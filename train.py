import torch
import time
import copy
import os
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, mixup_fn, device, config, logger):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mixup_fn = mixup_fn
        self.device = device
        self.config = config
        self.logger = logger

        self.scaler = GradScaler()
        self.best_accuracy = 0.0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def is_main_process(self):
        """
        Determines if the current process is the main process.
        Returns:
            bool: True if main process, False otherwise.
        """
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()

        running_loss = 0.0
        running_cls_loss = 0.0
        running_coh_loss = 0.0
        running_eval_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = [label.to(self.device, non_blocking=True) for label in labels]  # List of labels per stage

            # Apply Mixup if enabled
            if self.mixup_fn is not None:
                images, labels = self.mixup_fn(images, labels[-1])  # Assuming soft labels are only for the last stage
                # If you have soft labels for all stages, adjust accordingly

            self.optimizer.zero_grad()

            with autocast():
                # Forward pass through the model
                logits, P_t, features, similarities = self.model(images)  # logits: list of [batch, num_classes_t]

                # Compute Losses
                L_cls, L_coh, L_eval = self.loss_fn(logits, labels, similarities)

                L_total = L_cls + L_coh + self.config.LOSS.LAMBDA_EVAL * L_eval

            # Backward pass and optimization
            self.scaler.scale(L_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step (if step-based)
            if self.scheduler is not None and self.config.SCHEDULER.NAME.lower() == 'step':
                self.scheduler.step()

            # Accumulate losses
            running_loss += L_total.item()
            running_cls_loss += L_cls.item()
            running_coh_loss += L_coh.item()
            running_eval_loss += L_eval.item()

            if batch_idx % self.config['LOG_INTERVAL'] == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], "
                                 f"Loss: {L_total.item():.4f}, L_cls: {L_cls.item():.4f}, "
                                 f"L_coh: {L_coh.item():.4f}, L_eval: {L_eval.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_cls_loss = running_cls_loss / len(train_loader)
        epoch_coh_loss = running_coh_loss / len(train_loader)
        epoch_eval_loss = running_eval_loss / len(train_loader)

        self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], "
                         f"Avg Loss: {epoch_loss:.4f}, Avg L_cls: {epoch_cls_loss:.4f}, "
                         f"Avg L_coh: {epoch_coh_loss:.4f}, Avg L_eval: {epoch_eval_loss:.4f}")

        return epoch_loss

    def validate(self, val_loader, epoch):
        self.model.eval()

        running_loss = 0.0
        running_cls_loss = 0.0
        running_coh_loss = 0.0
        running_eval_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(self.device, non_blocking=True)
                labels = [label.to(self.device, non_blocking=True) for label in labels]

                # Forward pass through the model
                logits, P_t, features, similarities = self.model(images)

                # Compute Losses
                L_cls, L_coh, L_eval = self.loss_fn(logits, labels, similarities)

                L_total = L_cls + L_coh + self.config.LOSS.LAMBDA_EVAL * L_eval

                # Accumulate losses
                running_loss += L_total.item()
                running_cls_loss += L_cls.item()
                running_coh_loss += L_coh.item()
                running_eval_loss += L_eval.item()

                # Calculate accuracy (assuming labels are class indices)
                preds = [torch.argmax(p, dim=1) for p in logits]
                correct += (preds[-1] == labels[-1]).sum().item()
                total += labels[-1].size(0)

        epoch_loss = running_loss / len(val_loader)
        epoch_cls_loss = running_cls_loss / len(val_loader)
        epoch_coh_loss = running_coh_loss / len(val_loader)
        epoch_eval_loss = running_eval_loss / len(val_loader)
        accuracy = 100 * correct / total

        self.logger.info(f"Validation Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], "
                         f"Avg Loss: {epoch_loss:.4f}, Avg L_cls: {epoch_cls_loss:.4f}, "
                         f"Avg L_coh: {epoch_coh_loss:.4f}, Avg L_eval: {epoch_eval_loss:.4f}, "
                         f"Accuracy: {accuracy:.2f}%")

        # Check if this is the best model so far
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            if self.is_main_process():
                best_model_path = os.path.join(self.config.LOG_DIR, 'best_model.pt')
                torch.save(self.best_model_wts, best_model_path)
                self.logger.info(f"New best model saved with accuracy: {accuracy:.2f}% at {best_model_path}")

        return epoch_loss, accuracy

    def train(self, train_loader, val_loader, start_epoch=0):
        best_accuracy = self.best_accuracy
        best_model_wts = self.best_model_wts

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            epoch_start = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_accuracy = self.validate(val_loader, epoch)
            epoch_end = time.time()

            self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Time: {epoch_end - epoch_start:.2f}s")

            # Save checkpoint after each epoch
            if self.is_main_process():
                checkpoint_path = os.path.join(self.config.LOG_DIR, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Checkpoint saved at {checkpoint_path}")

            # Scheduler step (if not step-based and scheduler is epoch-based)
            if self.scheduler is not None and self.config.SCHEDULER.NAME.lower() != 'step':
                self.scheduler.step()

        # Load best model weights at the end of training
        self.model.load_state_dict(best_model_wts)
        self.logger.info(f"Training complete. Best Validation Accuracy: {best_accuracy:.2f}%")

import torch
import time
import copy
import os
from torch.cuda.amp import autocast, GradScaler

from utils import is_main_process


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, mixup_fn, device, config, logger, scaler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mixup_fn = mixup_fn
        self.device = device
        self.config = config
        self.logger = logger

        if scaler == None:
            self.scaler = GradScaler()
        else:
            self.scaler = scaler

        self.best_accuracy = 0.0

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()

        running_loss = 0.0
        running_mero_loss = 0.0
        running_base_loss = 0.0
        running_coh_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets["base"] = targets["base"].to(self.device, non_blocking=True)
            targets["mero"] = targets["mero"].to(self.device, non_blocking=True)

            # Apply Mixup if enabled
            # if self.mixup_fn is not None:
            #     images, labels = self.mixup_fn(images, labels[-1])

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.scaler.scale(losses["total_loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step (if step-based)
            if self.scheduler is not None and self.config.SCHEDULER.NAME.lower() == 'step':
                self.scheduler.step()

            # Accumulate losses
            running_loss += losses["total_loss"].item()
            running_mero_loss += losses["mero_loss"].item()
            running_base_loss += losses["base_loss"].item()
            running_coh_loss += losses["coh_loss"].item()

            self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], "
                                f"Loss: {losses['total_loss'].item():.4f}, L_mero: {losses['mero_loss'].item():.4f}, "
                                f"L_base: {losses['base_loss'].item():.4f}, L_coh: {losses['coh_loss'].item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_mero_loss = running_mero_loss / len(train_loader)
        epoch_base_loss = running_base_loss / len(train_loader)
        epoch_coh_loss = running_coh_loss / len(train_loader)

        self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], "
                         f"Avg Loss: {epoch_loss:.4f}, Avg L_cls: {epoch_mero_loss:.4f}, "
                         f"Avg L_coh: {epoch_base_loss:.4f}, Avg L_eval: {epoch_coh_loss:.4f}")

    def validate(self, val_loader, epoch):
        self.model.eval()

        running_loss = 0.0
        running_mero_loss = 0.0
        running_base_loss = 0.0
        running_coh_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(self.device, non_blocking=True)
                targets["base"] = targets["base"].to(self.device, non_blocking=True)
                targets["mero"] = targets["mero"].to(self.device, non_blocking=True)

                outputs = self.model(images)
                losses = self.criterion(outputs, targets)

                # Accumulate losses
                running_loss += losses["total_loss"].item()
                running_mero_loss += losses["mero_loss"].item()
                running_base_loss += losses["base_loss"].item()
                running_coh_loss += losses["coh_loss"].item()

                # Calculate accuracy (assuming labels are class indices)
                preds = outputs["base"].argmax(dim=1)
                correct += (preds == targets["base"]).sum().item()
                total += images.size(0)

        epoch_loss = running_loss / len(val_loader)
        epoch_mero_loss = running_mero_loss / len(val_loader)
        epoch_base_loss = running_base_loss / len(val_loader)
        epoch_coh_loss = running_coh_loss / len(val_loader)
        accuracy = 100 * correct / total

        self.logger.info(f"Validation Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], "
                         f"Avg Loss: {epoch_loss:.4f}, Avg L_mero: {epoch_mero_loss:.4f}, "
                         f"Avg L_base: {epoch_base_loss:.4f}, Avg L_coh: {epoch_coh_loss:.4f}, "
                         f"Accuracy: {accuracy:.2f}%")

        # Check if this is the best model so far
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            if is_main_process():
                save_state = {
                    'model': copy.deepcopy(self.model.state_dict()),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'max_accuracy': self.best_accuracy,
                    'scaler': self.scaler.state_dict(),
                    'epoch': epoch
                }

                best_model_path = os.path.join(os.path.join(self.config.LOG_DIR, self.config.DATASET.NAME), 'best_model.pt')
                torch.save(save_state, best_model_path)
                self.logger.info(f"New best model saved with accuracy: {accuracy:.2f}% at {best_model_path}")

    def train(self, train_loader, val_loader, start_epoch=0):
        best_accuracy = self.best_accuracy

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            epoch_start = time.time()
            self.train_one_epoch(train_loader, epoch)
            self.validate(val_loader, epoch)
            epoch_end = time.time()

            self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Time: {epoch_end - epoch_start:.2f}s")

            # Save checkpoint after each epoch
            if is_main_process():
                checkpoint_path = os.path.join(os.path.join(self.config.LOG_DIR, self.config.DATASET.NAME), f'checkpoint.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Checkpoint saved at {checkpoint_path}")

            # Scheduler step (if not step-based and scheduler is epoch-based)
            if self.scheduler is not None and self.config.SCHEDULER.NAME.lower() != 'step':
                self.scheduler.step()

        self.logger.info(f"Training complete. Best Validation Accuracy: {best_accuracy:.2f}%")

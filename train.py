import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
import copy
from torch.utils.tensorboard import SummaryWriter

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
    
    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
    
        running_loss = 0.0
        running_cls_loss = 0.0
        running_coh_loss = 0.0
        running_eval_loss = 0.0
    
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = [label.to(self.device, non_blocking=True) for label in labels]  # List of labels per stage
    
            # Apply Mixup
            # Mixup is applied only to the last stage labels (fine-grained)
            images, mixed_labels = self.mixup_fn(images, labels[-1])
            # Note: If Mixup needs to be applied to all stages, modify accordingly
    
            self.optimizer.zero_grad()
    
            with autocast():
                # Forward pass through the model
                logits, features, similarities = self.model(images)  # logits: list of [batch, num_classes_t]
    
                # Compute Losses
                L_cls, L_coh, L_eval = self.loss_fn(logits, labels, similarities)
    
                L_total = L_cls + L_coh + self.config['loss']['lambda_eval'] * L_eval
    
            # Backward pass and optimization
            self.scaler.scale(L_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
    
            # Scheduler step (if per batch)
            if self.scheduler['type'] == 'step':
                self.scheduler.step()
    
            # Accumulate losses
            running_loss += L_total.item()
            running_cls_loss += L_cls.item()
            running_coh_loss += L_coh.item()
            running_eval_loss += L_eval.item()
    
            if batch_idx % self.config['log_interval'] == 0 and self.config['rank'] == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.config['num_epochs']}], Step [{batch_idx}/{len(train_loader)}], "
                                 f"Loss: {L_total.item():.4f}, L_cls: {L_cls.item():.4f}, "
                                 f"L_coh: {L_coh.item():.4f}, L_eval: {L_eval.item():.4f}")
    
        epoch_loss = running_loss / len(train_loader)
        epoch_cls_loss = running_cls_loss / len(train_loader)
        epoch_coh_loss = running_coh_loss / len(train_loader)
        epoch_eval_loss = running_eval_loss / len(train_loader)
    
        self.logger.info(f"Epoch [{epoch+1}/{self.config['num_epochs']}], "
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
                logits, features, similarities = self.model(images)
    
                # Compute Losses
                L_cls, L_coh, L_eval = self.loss_fn(logits, labels, similarities)
    
                L_total = L_cls + L_coh + self.config['loss']['lambda_eval'] * L_eval
    
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
    
        self.logger.info(f"Validation Epoch [{epoch+1}/{self.config['num_epochs']}], "
                         f"Avg Loss: {epoch_loss:.4f}, Avg L_cls: {epoch_cls_loss:.4f}, "
                         f"Avg L_coh: {epoch_coh_loss:.4f}, Avg L_eval: {epoch_eval_loss:.4f}, "
                         f"Accuracy: {accuracy:.2f}%")
    
        return epoch_loss, accuracy
    
    def train(self, train_loader, val_loader):
        best_accuracy = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
    
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_accuracy = self.validate(val_loader, epoch)
            epoch_end = time.time()
    
            self.logger.info(f"Epoch [{epoch+1}/{self.config['num_epochs']}], Time: {epoch_end - epoch_start:.2f}s")
    
            # Deep copy the model
            if val_accuracy > best_accuracy and self.config['rank'] == 0:
                best_accuracy = val_accuracy
                best_model_wts = copy.deepcopy(self.model.state_dict())
    
            # Scheduler step (if not per batch)
            if self.scheduler['type'] != 'step':
                self.scheduler.step(val_loss)
    
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
    
        self.logger.info(f"Training complete. Best Validation Accuracy: {best_accuracy:.2f}%")

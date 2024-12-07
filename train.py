import torch
import time
import copy
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils import is_main_process, AverageMeter, accuracy, reduce_tensor


class Trainer:
    def __init__(self, model, model_without_ddp, criterion, optimizer, scheduler, mixup_fn, device, config, logger, scaler=None):
        self.model = model
        self.model_without_ddp = model_without_ddp
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

        running_loss = AverageMeter()
        running_mero_loss = AverageMeter()
        running_base_loss = AverageMeter()
        running_coh_loss = AverageMeter()

        for batch_idx, (data, targets) in enumerate(train_loader):
            images = data["images"].to(self.device, non_blocking=True)
            embeds = data["embeds"].to(self.device, non_blocking=True)
            targets["base"] = targets["base"].to(self.device, non_blocking=True)
            targets["mero"] = targets["mero"].to(self.device, non_blocking=True)

            # Apply Mixup if enabled
            if self.mixup_fn is not None:
                images, labels = self.mixup_fn(images, labels[-1])

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.AMP_ENABLE):
                outputs = self.model(images, embeds)
                losses = self.criterion(outputs, targets)

            if isinstance(self.scaler, GradScaler):
                self.scaler.scale(losses["total_loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.scaler(losses["total_loss"], self.optimizer, clip_grad=5.0, parameters=self.model.parameters(), 
                            create_graph=hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order)

            self.scheduler.step_update(epoch * len(train_loader) + batch_idx)

            # Accumulate losses
            running_loss.update(losses["total_loss"].item(), images.size(0))
            running_mero_loss.update(losses["mero_loss"].item(), images.size(0))
            running_base_loss.update(losses["base_loss"].item(), images.size(0))
            running_coh_loss.update(losses["coh_loss"].item(), images.size(0))

            if is_main_process():
                self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], "
                                 f"Loss: {losses['total_loss'].item():.4f}, L_mero: {losses['mero_loss'].item():.4f}, "
                                 f"L_base: {losses['base_loss'].item():.4f}, L_coh: {losses['coh_loss'].item():.4f}")

        if is_main_process():
            self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], "
                             f"Avg Loss: {running_loss.avg:.4f}, Avg L_mero: {running_mero_loss.avg:.4f}, "
                             f"Avg L_base: {running_base_loss.avg:.4f}, Avg L_coh: {running_coh_loss.avg:.4f}")

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()

        running_loss = AverageMeter()
        running_mero_loss = AverageMeter()
        running_base_loss = AverageMeter()
        running_coh_loss = AverageMeter()
        mero_accuracy_meter = AverageMeter()
        base_accuracy_meter = AverageMeter()

        for batch_idx, (data, targets) in enumerate(tqdm(val_loader)):
            images = data["images"].to(self.device, non_blocking=True)
            embeds = data["embeds"].to(self.device, non_blocking=True)
            targets["base"] = targets["base"].to(self.device, non_blocking=True)
            targets["mero"] = targets["mero"].to(self.device, non_blocking=True)

            outputs = self.model(images, embeds)
            losses = self.criterion(outputs, targets)

            acc = accuracy(outputs, targets)
            acc = reduce_tensor(acc)
            losses = reduce_tensor(losses)

            # Accumulate losses
            running_loss.update(losses["total_loss"].item(), images.size(0))
            running_mero_loss.update(losses["mero_loss"].item(), images.size(0))
            running_base_loss.update(losses["base_loss"].item(), images.size(0))
            running_coh_loss.update(losses["coh_loss"].item(), images.size(0))
            mero_accuracy_meter.update(acc["mero"].item(), images.size(0))
            base_accuracy_meter.update(acc["base"].item(), images.size(0))

        if is_main_process():
            self.logger.info(f"Avg Loss: {running_loss.avg:.4f}, Avg L_mero: {running_mero_loss.avg:.4f}, "
                             f"Avg L_base: {running_base_loss.avg:.4f}, Avg L_coh: {running_coh_loss.avg:.4f}, "
                             f"Mero accuracy: {mero_accuracy_meter.avg:.2f}%, Base accuracy: {base_accuracy_meter.avg:.2f}%")

        return base_accuracy_meter.avg

    def train(self, train_loader, val_loader, start_epoch=0):
        best_accuracy = self.best_accuracy

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            epoch_start = time.time()

            self.train_one_epoch(train_loader, epoch)

            val_acc = self.validate(val_loader)

            # Check if this is the best model so far
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                if is_main_process():
                    save_state = {
                        'model': copy.deepcopy(self.model_without_ddp.state_dict()),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'max_accuracy': self.best_accuracy,
                        'scaler': self.scaler.state_dict(),
                        'epoch': epoch
                    }
                    best_model_path = os.path.join(os.path.join(self.config.LOG_DIR, self.config.DATASET.NAME), 'best_model.pt')
                    torch.save(save_state, best_model_path)
                    self.logger.info(f"New best model saved with accuracy: {val_acc:.2f}% at {best_model_path}")

            epoch_end = time.time()

            if is_main_process():
                self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Time: {epoch_end - epoch_start:.2f}s")

            # Save checkpoint after each epoch
            if is_main_process():
                checkpoint_path = os.path.join(os.path.join(self.config.LOG_DIR, self.config.DATASET.NAME), f'checkpoint.pt')
                torch.save(self.model_without_ddp.state_dict(), checkpoint_path)
                self.logger.info(f"Checkpoint saved at {checkpoint_path}")

        if is_main_process():
            self.logger.info(f"Training complete. Best Validation Accuracy: {best_accuracy:.2f}%")

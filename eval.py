import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, loss_fn, device, config, logger):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.logger = logger
    
    def evaluate(self, test_loader):
        self.model.eval()
    
        running_loss = 0.0
        running_cls_loss = 0.0
        running_coh_loss = 0.0
        running_eval_loss = 0.0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
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
    
        epoch_loss = running_loss / len(test_loader)
        epoch_cls_loss = running_cls_loss / len(test_loader)
        epoch_coh_loss = running_coh_loss / len(test_loader)
        epoch_eval_loss = running_eval_loss / len(test_loader)
        accuracy = 100 * correct / total
    
        self.logger.info(f"Test Epoch [{self.config['num_epochs']}], "
                         f"Avg Loss: {epoch_loss:.4f}, Avg L_cls: {epoch_cls_loss:.4f}, "
                         f"Avg L_coh: {epoch_coh_loss:.4f}, Avg L_eval: {epoch_eval_loss:.4f}, "
                         f"Accuracy: {accuracy:.2f}%")
    
        return accuracy

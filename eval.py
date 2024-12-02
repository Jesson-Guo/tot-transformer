import torch
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, criterion, device, config, logger):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger

    def evaluate(self, test_loader):
        self.model.eval()

        running_loss = 0.0
        running_mero_loss = 0.0
        running_base_loss = 0.0
        running_coh_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
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

        epoch_loss = running_loss / len(test_loader)
        epoch_mero_loss = running_mero_loss / len(test_loader)
        epoch_base_loss = running_base_loss / len(test_loader)
        epoch_coh_loss = running_coh_loss / len(test_loader)
        accuracy = 100 * correct / total

        self.logger.info(f"Test Epoch [{self.config.NUM_EPOCHS}], "
                         f"Avg Loss: {epoch_loss:.4f}, Avg L_base: {epoch_mero_loss:.4f}, "
                         f"Avg L_base: {epoch_base_loss:.4f}, Avg L_coh: {epoch_coh_loss:.4f}, "
                         f"Accuracy: {accuracy:.2f}%")

        return accuracy

import torch

class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print("✅ Validation improved. Best model saved.")
        else:
            self.counter += 1
            print(f"⚠️ No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                print("⛔ Early stopping triggered!")
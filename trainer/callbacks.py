from transformers import TrainerCallback

class BestIoUCallback(TrainerCallback):
    def __init__(self):
        self.best_iou = -1.0  # Initialize with a low value

    def on_evaluate(self, args, state, control, **kwargs):
        current_iou = state.metrics["eval_iou"]
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            control.save_model()  # Save the model when IoU improves
            print(f"New best model saved with IoU: {current_iou:.4f}")
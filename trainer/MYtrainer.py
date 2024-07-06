from transformers import Trainer
from torch.utils.data import Dataset
import torch 
from sklearn.metrics import f1_score, jaccard_score

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    pred_logits = p.labels
    pred_probs = torch.sigmoid(torch.tensor(pred_logits))
    pred_labels = (pred_probs > 0.5).float().numpy()

    true_labels = p.labels

    # Flatten arrays
    pred_labels = pred_labels.flatten()
    true_labels = true_labels.flatten()

    dice = f1_score(true_labels, pred_labels, average='binary')
    iou = jaccard_score(true_labels, pred_labels, average='binary')

    return {
        "dice": dice,
        "iou": iou,
    }
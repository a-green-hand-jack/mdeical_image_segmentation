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


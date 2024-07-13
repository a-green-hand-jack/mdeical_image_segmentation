import torch

class AMOSDataCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        volumes = [sample["volume"] for sample in batch]
        labels = [sample["label"] for sample in batch]

        # Stack images and masks along batch dimension
        volumes = torch.stack(volumes, dim=0)
        labels = torch.stack(labels, dim=0)

        return {"volumes": volumes, "labels": labels}
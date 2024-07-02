import torch

class DRIVEDataCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        images = [sample["image"] for sample in batch]
        masks = [sample["mask"] for sample in batch]

        # Stack images and masks along batch dimension
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)

        return {"images": images, "masks": masks}
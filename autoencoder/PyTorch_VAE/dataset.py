import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from PIL import Image
import numpy as np
import yaml

path_yaml = 'scripts/detection/models/vae_celeba.yaml'


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class Dataset_from_list(Dataset):
    def __init__(self, transform, images, annotations, pathtoimg):
        self.images = images
        self.annotations = annotations
        self.transform = transform
        self.pathtoimg = pathtoimg
        self.id = {k: v for v, k in images}
        with open(path_yaml, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        self.patch_size = config["data_params"]['patch_size']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        box, id = self.annotations[idx]
        img = self.id[id]
        path_img = os.path.join(self.pathtoimg, img)
        img = Image.open(path_img)
        if img.layers == 1:
            img = img.convert('RGB')
        img = np.array(img)
        if not box is None:
            box = [int(x) for x in box]
            image = img[box[1]: box[1]+box[3], box[0]: box[0]+box[2]]
        else:
            image = img
        pilim = Image.fromarray(image)
        pilim = pilim.resize((self.patch_size, self.patch_size))

        # pilim.save('/home/neptun/PycharmProjects/datasets/coco/ds/{}.jpg'.format(idx))

        if self.transform:
            image = self.transform(pilim)
        return image, idx


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.images = kwargs['images']
        self.annotations = kwargs['annotations']
        self.pathtoimg = kwargs['pathtoimg']

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([
                                              # transforms.ToPILImage(),
                                              transforms.RandomHorizontalFlip(),
                                              # transforms.CenterCrop(148),
                                              # transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([
                                            # transforms.ToPILImage(),
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.CenterCrop(148),
                                            # transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])

        self.train_dataset = Dataset_from_list(transform=train_transforms, images=self.images,
                                               annotations=self.annotations, pathtoimg=self.pathtoimg)
        self.val_dataset = Dataset_from_list(transform=val_transforms, images=self.images,
                                               annotations=self.annotations, pathtoimg=self.pathtoimg)
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            # num_workers=self.num_workers,
            shuffle=True,
            # pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            # num_workers=self.num_workers,
            shuffle=False,
            # pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            # num_workers=self.num_workers,
            shuffle=True,
            # pin_memory=self.pin_memory,
        )
     
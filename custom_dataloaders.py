import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from pytorch_classifier.image import Rescale, normalize, RandomColorDistort
from pytorch_classifier.dataset import FolderDataset, FolderAugmentedDataset, FolderAugmentedTenCropDataset


def get_simple_dataloader(data_path, shuffle=False, batch_size=16):
    trans = transforms.Compose([
        # transforms.Resize((224,224)),
        # transforms.CenterCrop((224,224)),
        Rescale((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    return DataLoader(FolderDataset(data_path, trans), shuffle=shuffle, batch_size=batch_size, num_workers=16)


def get_ensemble_of_augmentations_dataloader(data_path, shuffle=False, batch_size=16):
    trans = transforms.Compose([
        Rescale((264, 264)),
        # transforms.RandomResizedCrop(size=224, scale=(0.8, 1), ratio=(1, 1)),
        transforms.RandomResizedCrop(size=224),
        RandomColorDistort(),
        # Rescale((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    return DataLoader(FolderAugmentedDataset(data_path, trans, 7), shuffle=shuffle, batch_size=batch_size, num_workers=16)


def get_ten_crop_dataloader(data_path, shuffle=False, batch_size=16):
    trans = transforms.Compose([
        Rescale((240, 240)),
        # ResizeForTenCrop(224, 226),
        transforms.TenCrop((224, 224)),
        lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(), normalize])(crop) for crop in crops])
    ])

    return DataLoader(FolderAugmentedTenCropDataset(data_path, trans), shuffle=shuffle, batch_size=batch_size, num_workers=16)

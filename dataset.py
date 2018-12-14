import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.folder = folder
        self.images_path = [os.path.join(folder, file_name) for file_name in os.listdir(folder)]
        # self.images_path = self.images_path[:500] + self.images_path[-500:]
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_path = self.images_path[item]
        img = Image.open(image_path)
        img = self.transform(img)
        return image_path, img


class FolderAugmentedDataset(FolderDataset):
    def __init__(self, folder, transforms_and_augmentations, num_results):
        super(FolderAugmentedDataset, self).__init__(folder, transforms_and_augmentations)
        self.transforms_and_augmentations = transforms_and_augmentations
        self.num_results = num_results

    def __getitem__(self, item):
        image_path = self.images_path[item]
        img = Image.open(image_path)
        result_images = []
        for i in range(self.num_results):
            result_images.append(self.transforms_and_augmentations(img).unsqueeze(0))
        result_images = torch.cat(result_images)
        return image_path, result_images


class FolderAugmentedTenCropDataset(FolderDataset):
    def __init__(self, folder, transforms_and_augmentations):
        super(FolderAugmentedTenCropDataset, self).__init__(folder, transforms_and_augmentations)
        self.transforms_and_augmentations = transforms_and_augmentations

    def __getitem__(self, item):
        image_path = self.images_path[item]
        img = Image.open(image_path)
        result_images=self.transforms_and_augmentations(img)
        return image_path, result_images

class CatsAndDogsDataSet(FolderDataset):
    def __getitem__(self, item):
        image_path = self.images_path[item]
        img = Image.open(image_path)
        img = self.transform(img)
        target = os.path.basename(image_path)[:3] == 'cat'
        return image_path, img, target


IMAGENET_DOG_CLASSES = list(range(151,276))
IMAGENET_CAT_CLASSES = list(range(281,286))

CATS_AND_DOGS_CLASSES = {0:IMAGENET_DOG_CLASSES, 1:IMAGENET_CAT_CLASSES}

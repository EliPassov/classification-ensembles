import numpy as np
import tqdm
import os

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms, datasets, models

from pytorch_classifier.dataset import FolderDataset, FolderAugmentedDataset, FolderAugmentedTenCropDataset, CATS_AND_DOGS_CLASSES, CatsAndDogsDataSet
from pytorch_classifier.image import Rescale, RandomColorDistort, normalize
from pytorch_classifier.ensemble_net import majority_vote
from pytorch_classifier.net_utils import get_partial_classifier


class Evaluator():
    def __init__(self, net, data_loader):
        self.data_loader = data_loader
        self.net = net

    def eval(self):
        self.net.eval()
        with torch.no_grad():
            result = self.eval_inner()
        self.net.train()
        return result

    def eval_inner(self):
        raise NotImplementedError('Method should be implemented in inheriting class')


class SimpleEvaluator(Evaluator):
    def eval_inner(self):
        results = {}
        for image_paths, image_batch in self.data_loader:
            image_batch = image_batch.cuda()
            predictions = self.net(image_batch)
            for i in range(len(image_paths)):
                results[image_paths[i]] = predictions[i]
            # print(image_paths, predictions)
        return results


class EnsembleEvaluator(Evaluator):
    def __init__(self, net, data_loader, ensemble_module):
        super(EnsembleEvaluator, self).__init__(net, data_loader)
        self.ensemble_module = ensemble_module

    def eval_inner(self):
        results = {}
        for image_paths, images_batches in self.data_loader:
            images_batches = images_batches.cuda()
            s = images_batches.shape
            images_batches=images_batches.view(s[0]*s[1],s[2],s[3],s[4])
            # ensemble_size = len(image_batches)
            # s = image_batches[0].shape
            # image_batches = [image_batch.cuda() for image_batch in image_batches]
            # unified_image_batches = torch.cat(image_batches, 0).view(s[0], ensemble_size,s[1],s[2],s[3]).permute(1,0,2,3,4).contiguous()
            predictions = self.net(images_batches)
            predictions = np.array(predictions).reshape(s[0],s[1])
            for i in range(len(image_paths)):
                results[image_paths[i]] = self.ensemble_module(predictions[i])
            # print(image_paths, predictions)
        return results


def evaluate_simple(data_path):
    base_data_transform = transforms.Compose([
        # transforms.Resize((224,224)),
        # transforms.CenterCrop((224,224)),
        Rescale((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    data_loader = tqdm.tqdm(DataLoader(FolderDataset(data_path, base_data_transform), shuffle=False, batch_size=16, num_workers=16))
    return SimpleEvaluator(net, data_loader).eval()


def evaluate_ensemble_of_augmentations(data_path):
    augmented_data_transform = transforms.Compose([
        # Rescale((264,264)),
        # transforms.RandomResizedCrop(size=224, scale=(0.8, 1), ratio=(1, 1)),
        # transforms.RandomResizedCrop(size=224),
        RandomColorDistort(),
        Rescale((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    data_loader = tqdm.tqdm(DataLoader(FolderAugmentedDataset(data_path, augmented_data_transform, 7), shuffle=False, batch_size=16, num_workers=16))
    return EnsembleEvaluator(net, data_loader, majority_vote).eval()


def evaluate_ensemble_of_augmentations_ten_crop(data_path):
    augmented_data_transform = transforms.Compose([
        Rescale((240,240)),
        transforms.TenCrop((224,224)),
        lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),normalize])(crop) for crop in crops])
    ])
    data_loader = tqdm.tqdm(DataLoader(FolderAugmentedTenCropDataset(data_path, augmented_data_transform), shuffle=False, batch_size=16, num_workers=16))
    return EnsembleEvaluator(net, data_loader, majority_vote).eval()


def check_predictions_cats_and_dogs(predictions):
    corrects = 0
    for img_path, prediction in predictions.items():
        if prediction == 1 and os.path.basename(img_path)[:3] == 'cat' or \
                prediction == 0 and os.path.basename(img_path)[:3] == 'dog':
            corrects += 1

    return corrects/len(predictions)


if __name__=='__main__':
    net=get_partial_classifier('squeezenet1_0')
    # resnet = models.resnet50(pretrained=True)
    # net = PartialClassesClassifier(resnet, CATS_AND_DOGS_CLASSES).cuda()

    eval_path = '/home/eli/Data/cats_vs_dogs/sub_eval'
    # train_path = '/home/eli/Data/COCO/coco/images/val2014'

    # predictions = evaluate_simple(eval_path)
    # predictions = evaluate_ensemble_of_augmentations(eval_path)
    predictions = evaluate_ensemble_of_augmentations_ten_crop(eval_path)

    print(check_predictions_cats_and_dogs(predictions))
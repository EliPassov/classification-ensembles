import numpy as np
import tqdm
import os

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from pytorch_classifier.dataset import FolderDataset, FolderAugmentedDataset, FolderAugmentedTenCropDataset
from pytorch_classifier.ensemble_net import majority_vote, MajorityEnsembleModule, AverageEnsembleModule
from pytorch_classifier.net_utils import NetWithResult, get_partial_classifier
from pytorch_classifier.custom_dataloaders import *


class Evaluator():
    def __init__(self, net, data_loader):
        self.data_loader = data_loader
        self.net = net

    def eval(self):
        self.net.eval()
        with torch.no_grad():
            results = {}
            for image_paths, image_batch in self.data_loader:
                image_batch = image_batch.cuda()
                predictions = self.net(image_batch)
                for i in range(len(image_paths)):
                    results[image_paths[i]] = predictions[i]

        self.net.train()
        return results


def check_predictions_cats_and_dogs(predictions):
    corrects = 0
    for img_path, prediction in predictions.items():
        if prediction == 1 and os.path.basename(img_path)[:3] == 'cat' or \
                prediction == 0 and os.path.basename(img_path)[:3] == 'dog':
            corrects += 1

    return corrects/len(predictions)


if __name__=='__main__':
    #TODO: Change this to args

    #################################################
    ### get pre-trained net with imagenet classes ###
    #################################################
    # net = get_partial_classifier('squeezenet1_0')
    ### ensemble wrapper - optional
    # net = MajorityEnsembleModule(net, True)

    ##############################################
    ### OR: load trained transfer learning net ###
    ##############################################
    ### simple trained net
    # net = torch.load('/home/eli/test/cats_vs_dogs/squeezenet_super_mini_train_simple/net_e_199_score_0_9100.pt')
    ### augmented trained net
    net = torch.load('/home/eli/test/cats_vs_dogs/squeezenet_super_mini_train_from_scratch_with_auxilary2/net_e_171_score_0_9500.pt')
    ### wrap net:
    ### simple
    net = NetWithResult(net)
    ### or ensembles
    # net = MajorityEnsembleModule(net)
    # net = AverageEnsembleModule(net)

    eval_path = '/home/eli/Data/cats_vs_dogs/sub_eval'

    ############################################################
    ### choose loader (simple or ensemble, depending on net) ###
    ############################################################
    data_loader = get_simple_dataloader(eval_path)
    # data_loader = get_ensemble_of_augmentations_dataloader(eval_path)
    # data_loader = get_ten_crop_dataloader(eval_path)

    predictions = Evaluator(net, tqdm.tqdm(data_loader)).eval()
    print(check_predictions_cats_and_dogs(predictions))

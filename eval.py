import argparse
import numpy as np
import tqdm
import os

from pytorch_classifier.ensemble_net import get_ensemble_class
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


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_net", action="store", default=None)
    parser.add_argument("--trained_net_path", action="store", default=None)
    parser.add_argument("--ensemble", action="store", default=None)
    parser.add_argument("--eval_folder_path", action="store", default=None)
    return parser.parse_args()


def parse_args(args):
    assert args.eval_folder_path is not None, '--eval_folder_path must be provided for evaluation run'

    if args.pretrained_net is not None:
        net = get_partial_classifier(args.pretrained_net)
    elif args.trained_net_path is not None:
        net=torch.load(args.trained_net_path)
    else:
        raise ValueError('Must either provide --pretrained_net name (e.g. squeezenet1_0) or '
                         '--trained_net_path a transfer learning trained net file path')

    if args.ensemble is not None:
        if args.pretrained_net is not None:
            assert args.ensemble == 'majority', 'Can only apply majority ensemble for pretrained net'
        net = get_ensemble_class(args.ensemble)(net)
        data_loader = get_ten_crop_dataloader(args.eval_folder_path)
    else:
        if args.trained_net_path is not None:
            net = NetWithResult(net)
        data_loader = get_simple_dataloader(args.eval_folder_path)


    return net, data_loader

if __name__=='__main__':
    args = setup()
    net, data_loader = parse_args(args)

    predictions = Evaluator(net, tqdm.tqdm(data_loader)).eval()
    print(check_predictions_cats_and_dogs(predictions))

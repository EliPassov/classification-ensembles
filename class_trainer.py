import os
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from tensorboardX import SummaryWriter
import tqdm
from torch.autograd import Variable
from pytorch_classifier.net_utils import NetWithResult


from pytorch_classifier.dataset import FolderDataset, CatsAndDogsDataSet, CATS_AND_DOGS_CLASSES
from pytorch_classifier.image import normalize, Rescale
from pytorch_classifier.net_utils import create_partially_trainable_net
from pytorch_classifier.eval import Evaluator, check_predictions_cats_and_dogs, get_ten_crop_dataloader
from pytorch_classifier.ensemble_net import get_ensemble_class


base_data_transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor(),
    normalize
])


def train(net, backup_folder, train_loader, epochs_to_run, evaluator, loss_func, optimizer, auxiliary_net=None,
          auxiliary_loader=None):
    net = net.cuda()

    writer = SummaryWriter(os.path.join(backup_folder, 'log'))

    avg_loss = None

    if auxiliary_net is not None:
        auxiliary_net = auxiliary_net.cuda()
        it = iter(auxiliary_loader)

    for epoch in range(epochs_to_run):
        for batch_id, (img_path, data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())

            optimizer.zero_grad()

            if auxiliary_net is not None:
                try:
                    img_path2, data2 = next(it)
                except StopIteration:
                    it = iter(auxiliary_loader)
                    img_path2, data2 = next(it)

                data2 = Variable(data2.cuda())
                target2 = torch.LongTensor(auxiliary_net(data2)).cuda()
                # assuming the first image in the augmented data loader is a non-augmented image
                data=torch.cat([data,data2[:, 0, ...]])
                target=torch.cat([target, target2])

            output = net(data)

            loss = loss_func(output, target)

            if avg_loss is None:
                avg_loss = loss.item()
            else:
                avg_loss = 0.9*avg_loss + 0.1*loss.item()

            print('{}, {}: {:.4f}, {:.4f}'.format(epoch, batch_id, loss.item(), avg_loss))
            if batch_id % 10 == 1:
                writer.add_scalar('train/loss',avg_loss, batch_id+epoch*train_loader.__len__())

            loss.backward()
            optimizer.step()
        predictions = evaluator.eval()
        eval_score = check_predictions_cats_and_dogs(predictions)
        torch.save(net, os.path.join(backup_folder, 'net_e_{}_score_{}.pt'.format(epoch, '{:.4f}'.format(eval_score).replace('.','_'))))
        print(eval_score)
        writer.add_scalar('eval/score', eval_score, epoch)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup_folder", action="store", default=None)
    parser.add_argument("--batch_size", action="store", default=16)
    parser.add_argument("--pretrained_net", action="store", default=None)
    parser.add_argument("--num_classes", action="store", default=None)
    parser.add_argument("--trained_net_path", action="store", default=None)
    parser.add_argument("--auxiliary_net_path", action="store", default=None)
    parser.add_argument("--train_data_path", action="store", default=None)
    parser.add_argument("--auxiliary_data_path", action="store", default=None)
    parser.add_argument("--ensemble", action="store", default=None)
    parser.add_argument("--eval_data_path", action="store", default=None)
    return parser.parse_args()


def parse_args(args):
    assert args.backup_folder is not None
    backup_folder = args.backup_folder

    batch_size = int(args.batch_size)

    if args.trained_net_path is not None:
        net = torch.load(args.trained_net_path)
    elif args.pretrained_net is not None:
        assert args.num_classes is not None
        net = create_partially_trainable_net(getattr(models, args.pretrained_net)(pretrained=True), int(args.num_classes))
    else:
        raise ValueError('Either --trained_net_path or --pretrained_net(with --num_classes) must be provided')

    train_batch_size = batch_size if args.auxiliary_net_path is None else batch_size // 2

    train_loader = DataLoader(CatsAndDogsDataSet(args.train_data_path, base_data_transform), shuffle=True,
                              batch_size=train_batch_size, num_workers=16)
    test_loader = DataLoader(FolderDataset(args.eval_data_path, base_data_transform), shuffle=False,
                             batch_size=batch_size, num_workers=16)
    evaluator = Evaluator(NetWithResult(net), test_loader)


    auxiliary_net, auxiliary_loader = None, None
    if args.auxiliary_net_path is not None:
        assert args.auxiliary_data_path is not None
        assert args.ensemble is not None
        auxiliary_net = torch.load(args.auxiliary_net_path)
        auxiliary_net = get_ensemble_class(args.ensemble)(auxiliary_net)
        auxiliary_loader = get_ten_crop_dataloader(args.auxiliary_data_path, shuffle=True, batch_size=train_batch_size)

    return net, batch_size, backup_folder, train_loader, evaluator, auxiliary_net, auxiliary_loader


if __name__=='__main__':
    args = setup()
    net, batch_size, backup_folder, train_loader, evaluator, auxiliary_net, auxiliary_loader = parse_args(args)

    loss_func = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    momentum = 0.95
    decay = 0.0005

    optimizer_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.SGD(optimizer_parameters, lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)

    train(net, backup_folder, train_loader, epochs_to_run=200, evaluator=evaluator, loss_func=loss_func,
          optimizer=optimizer, auxiliary_net = auxiliary_net, auxiliary_loader=auxiliary_loader)



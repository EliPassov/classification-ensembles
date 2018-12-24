import os
from PIL import Image
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
from pytorch_classifier.net_utils import get_partial_classifier, create_partially_trainable_net, PartialClassesClassifier
from pytorch_classifier.eval import Evaluator, check_predictions_cats_and_dogs, get_ten_crop_dataloader
from pytorch_classifier.ensemble_net import MajorityEnsembleModule


base_data_transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor(),
    normalize
])


def train(net, backup_folder, train_loader, epochs_to_run, evaluator, loss_func, learning_rate, batch_size, momentum,
          decay, auxilary_net=None, auxiliary_loader=None):
    net = net.cuda()
    optimizer_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.SGD(optimizer_parameters, lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)

    writer = SummaryWriter(os.path.join(backup_folder, 'log'))

    avg_loss = None

    if auxilary_net is not None:
        it = iter(auxiliary_loader)

    for epoch in range(epochs_to_run):
        for batch_id, (img_path, data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())

            optimizer.zero_grad()

            if auxilary_net is not None:
                try:
                    img_path2, data2 = next(it)
                except StopIteration:
                    it = iter(auxiliary_loader)
                    img_path2, data2 = next(it)

                data2 = Variable(data2.cuda())
                target2 = torch.LongTensor(auxilary_net(data2)).cuda()
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


if __name__=='__main__':
    net = create_partially_trainable_net(models.squeezenet1_0(pretrained=True), 2)
    # net = torch.load('/home/eli/test/cats_vs_dogs/squeezenet_super_mini_train_simple/net_e_199_score_0_9100.pt')

    train_path = '/home/eli/Data/cats_vs_dogs/super_mini_train1'
    auxiliary_path = '/home/eli/Data/cats_vs_dogs/super_mini_train2'
    eval_path = '/home/eli/Data/cats_vs_dogs/super_mini_eval'

    backup_folder = '/home/eli/test/cats_vs_dogs/squeezenet_super_mini_train_from_scratch_with_auxilary2'

    auxilary_net, auxiliary_loader = None, None
    auxiliary_loader = None

    # Optional: train with unsupervised data tagging using auxiliary net (same or different) with ensemble inference

    # auxilary_net = net
    auxilary_net = torch.load('/home/eli/test/cats_vs_dogs/squeezenet_super_mini_train_simple/net_e_199_score_0_9100.pt')
    auxilary_net = MajorityEnsembleModule(auxilary_net).cuda()

    loss_func = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    batch_size = 16
    momentum = 0.95
    decay = 0.0005

    train_batch_size = batch_size if auxilary_net is None else batch_size // 2

    train_loader = DataLoader(CatsAndDogsDataSet(train_path, base_data_transform), shuffle=True,
                              batch_size=train_batch_size, num_workers=16)
    test_loader = DataLoader(FolderDataset(eval_path, base_data_transform), shuffle=False, batch_size=batch_size,
                             num_workers=16)
    evaluator = Evaluator(NetWithResult(net), test_loader)

    if auxilary_net is not None:
        auxiliary_loader = get_ten_crop_dataloader(auxiliary_path, shuffle=True, batch_size=train_batch_size)


    train(net, backup_folder, train_loader, epochs_to_run=200, evaluator=evaluator, loss_func=loss_func,
          learning_rate=learning_rate, batch_size=batch_size, momentum=momentum, decay=decay,
          auxilary_net = auxilary_net, auxiliary_loader=auxiliary_loader)



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


from pytorch_classifier.dataset import FolderDataset, CatsAndDogsDataSet, CATS_AND_DOGS_CLASSES
from pytorch_classifier.image import normalize, Rescale
from pytorch_classifier.net_utils import get_partial_classifier, create_partially_trainable_net, PartialClassesClassifier
from pytorch_classifier.eval import SimpleEvaluator, check_predictions_cats_and_dogs


base_data_transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor(),
    normalize
])


def train(net, backup_folder, train_path, epochs_to_run, evaluator, loss_func, learning_rate, batch_size, momentum, decay):
    net = net.cuda()
    optimizer_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.SGD(optimizer_parameters, lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)

    writer = SummaryWriter(os.path.join(backup_folder, 'log'))
    train_loader = DataLoader(CatsAndDogsDataSet(train_path, base_data_transform), shuffle=True, batch_size=16, num_workers=16)

    avg_loss = None

    for epoch in range(epochs_to_run):
        for batch_id, (img_path, data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())

            optimizer.zero_grad()

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


class NetWithResult(nn.Module):
    def __init__(self, net):
        super(NetWithResult, self).__init__()
        self.net = net

    def forward(self, x):
        x= self.net(x)
        return torch.argmax(x,-1)

if __name__=='__main__':
    net = models.squeezenet1_0(pretrained=True)

    # net=get_partial_classifier('squeezenet1_0')
    net = create_partially_trainable_net(net, 2)
    # net = PartialClassesClassifier(net, CATS_AND_DOGS_CLASSES)

    # resnet = models.resnet50(pretrained=True)
    # net = PartialClassesClassifier(resnet, CATS_AND_DOGS_CLASSES).cuda()

    train_path = '/home/eli/Data/cats_vs_dogs/mini_train1'
    eval_path = '/home/eli/Data/cats_vs_dogs/mini_eval'

    backup_folder = '/home/eli/test/cats_vs_dogs/squeezenet_mini_train'
    # train_path = '/home/eli/Data/COCO/coco/images/val2014'

    test_loader = DataLoader(FolderDataset(eval_path, base_data_transform), shuffle=False, batch_size=32, num_workers=16)
    eval_net = NetWithResult(net)
    evaluator = SimpleEvaluator(eval_net, test_loader)
    loss_func = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    batch_size = 16
    momentum = 0.95
    decay = 0.0005
    train(net, backup_folder, train_path, epochs_to_run=50, evaluator=evaluator, loss_func=loss_func, learning_rate=learning_rate,
          batch_size=batch_size, momentum=momentum, decay=decay)

    # predictions = evaluate_simple(train_path)


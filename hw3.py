# Adam Craig
# Deep Learning
# HW3
# Pretrained ResNet50 and VGG19 on MNIST
# The train loop was taken from pytorch's tutorial

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import time
import sys

cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class N_Dataset(Dataset):
    def __init__(self, N: int, root, train, download, transform):
        self.dataset = datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )
        self._len = N*10
        self.counts = { i : 0 for i in range(10)}
        self.array = list()
        self._populate(N)

    def _populate(self, N: int):
        # populate the N-subset that will be used
        for idx in range(self.dataset.__len__()):
            X, label = self.dataset.__getitem__(idx)
            if self.counts[label] < N:
                Y = torch.vstack([X, X, X])
                self.array.append((Y, label))
                self.counts[label] += 1

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.array[idx]

class Reshape_Dataset(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )
        self.array = list()
        self._populate()

    def _populate(self):
        for idx in range(self.dataset.__len__()):
            X, label = self.dataset.__getitem__(idx)
            Y = torch.vstack([X, X, X])
            self.array.append((Y,label))

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.array[idx]

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

# pass in two arguments in the command line, a and b,
# for example python3 hw3.py 2 30
# would finetune a VGG19 model with a 30*10 resource regime
def main():
    if len(sys.argv) == 1:
        print(
            "Please enter an argument for which part to complete. Ex: `python hw3.py 2`"
        )
        return
    flag = int(sys.argv[1])

    if flag == 1:
        size = int(sys.argv[2])
        model_conv = torchvision.models.resnet50(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False
        
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 10)
        
        model_conv = model_conv.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

        print(f"**********************************************************\nSize: 10*{size}")
        training_dataset = N_Dataset(size, root="data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]))
        testing_dataset = Reshape_Dataset(root="data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])) 
        dataloaders = {
            "train" : DataLoader(training_dataset, batch_size=4, shuffle=True, num_workers=0),
            "val" : DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=0),
        } 
        sizes = {
            "train" : training_dataset.__len__(),
            "val" : testing_dataset.__len__(),
        }

        train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, sizes, num_epochs=12)
    
    if flag == 2:
        size = int(sys.argv[2])
        model_conv = torchvision.models.vgg19(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False
        
        num_ftrs = 512*7*7 # defined at https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg19
        model_conv.classifier = nn.Linear(num_ftrs, 10)
        
        model_conv = model_conv.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

        print(f"**********************************************************\nSize: 10*{size}")

        # The VGG19 is so large (~150M params) that we must limit the size of the validation
        # set -- the main bottleneck of finetuning in this low-resource experiment
        # This is done out of necessity...
        training_dataset = N_Dataset(size, root="data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]))
        testing_dataset = N_Dataset(100, root="data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])) 
        dataloaders = {
            "train" : DataLoader(training_dataset, batch_size=4, shuffle=True, num_workers=0),
            "val" : DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=0),
        } 
        sizes = {
            "train" : training_dataset.__len__(),
            "val" : testing_dataset.__len__(),
        }

        train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, sizes, num_epochs=12)
main()
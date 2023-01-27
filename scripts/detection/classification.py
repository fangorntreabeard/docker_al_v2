from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import shutil
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # print(f'Epoch {epoch}/{num_epochs - 1}')
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0
            f1_pred = []
            f1_true = []

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
                f1_true = f1_true + labels.data.tolist()
                f1_pred = f1_pred + preds.tolist()
                # running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = f1_score(f1_true, f1_pred)

            # print(f'{phase} Loss: {epoch_loss:.4f} f1: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, unlabeled_data, transform=None):
        self.img_labels = unlabeled_data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path)
        try:
            if image.layers == 1:
                image = image.convert('RGB')
        except:
            pass
            # print(idx)
        # image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, idx


def find_be_images(device, model, pathtoimg, unlabeled_data, datatransforms, class_names):
    model.eval()
    image_datasets = CustomImageDataset(pathtoimg, unlabeled_data, datatransforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=8,  shuffle=False, num_workers=4)
    new_images = []
    with torch.no_grad():
        for inputs, ind in dataloaders:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            where_be = torch.where(preds == class_names.index('be'))[0].tolist()
            for k in where_be:
                new_images.append(unlabeled_data[ind[k]])
    return new_images



def classification(device, pathtoimg, path_to_classes, images, annotations, unlabeled_data):

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # data_dir = os.path.join(pathtoimg, 'classification')
    for a in ['train', 'val']:
        for b in ['be', 'notbe']:
            path_cl = os.path.join(path_to_classes, a, b)
            files = os.listdir(path_cl)
            for f in files:
                os.remove(os.path.join(path_cl, f))
    dict_img = {k: n for n, k in images}
    imgid_be = []
    allid_img = []
    for row in annotations:
        imgid_be.append(row[1])
    for row in images:
        allid_img.append(row[1])
    imgid_not_be = list(set(allid_img) - set(imgid_be))
    train_be, val_be = train_test_split(imgid_be, test_size=0.2)
    train_notbe, val_notbe = train_test_split(imgid_not_be, test_size=0.2)
    for k in train_be:
        shutil.copyfile(os.path.join(pathtoimg, dict_img[k]), os.path.join(path_to_classes,
                                                                           'train', 'be', dict_img[k]))
    for k in val_be:
        shutil.copyfile(os.path.join(pathtoimg, dict_img[k]), os.path.join(path_to_classes,
                                                                           'val', 'be', dict_img[k]))
    for k in train_notbe:
        shutil.copyfile(os.path.join(pathtoimg, dict_img[k]), os.path.join(path_to_classes,
                                                                           'train', 'notbe', dict_img[k]))
    for k in val_notbe:
        shutil.copyfile(os.path.join(pathtoimg, dict_img[k]), os.path.join(path_to_classes,
                                                                           'val', 'notbe', dict_img[k]))


    image_datasets = {x: datasets.ImageFolder(os.path.join(path_to_classes, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           dataloaders, dataset_sizes, device, num_epochs=10)

    # visualize_model(model_ft)
    return find_be_images(device, model_ft, pathtoimg, unlabeled_data, data_transforms['val'], class_names)
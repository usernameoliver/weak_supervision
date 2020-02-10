'''
train the model given model name and dataset
train the model given model name and dataset
@Author: Degan Hao
@Date: Feb 04, 2020

'''
BASE_DIR = './'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import time
from sklearn import metrics
from pathlib import Path


'''
load data
'''

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.RandomResizedCrop(224),
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

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple[0], path)
        return tuple_with_path

def load_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model_name == 'densenet':
        model = models.densenet161(pretrained=True)
    elif model_name == 'resnext':
        model = models.resnext50_32x4d(pretrained=True)
    else:
        print('model not found!!!')
    return model


def train_model(model, model_name, criterion, optimizer, scheduler, num_epochs, data_dir, subset, flipped_label_dict):
    base_path =  BASE_DIR + 'evaluation/mmf/models/' 
    state_save_path = base_path + model_name +  'ddsm_state.pt'
    entire_model_save_path = base_path + model_name + 'ddsm_entire_model.pt'
    check_point_save_path = base_path + model_name + 'ddsm_checkpoint_model.pt'

    weight_file = Path(entire_model_save_path)

    '''
    if weight_file.is_file():
        model = torch.load(entire_model_save_path)
        return model
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=1) for x in ['train', 'val']} 
    dataset_sizes = {'train': len(image_datasets['train']) - len(subset),  'val': len(image_datasets['val']) }
    '''
    f = open(flipped_label_dict_path, "rb")
    flipped_label_dict = pickle.load(f)
    '''


    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0

    for epoch in range(num_epochs):
        '''
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        '''
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            pred = []
            y_label = []
            running_auc = 0.0
            for inputs, path in dataloaders[phase]:
                image_name = path[0].split('/')[-1] 
                if image_name in subset:
                    continue
                labels = torch.tensor([flipped_label_dict[image_name]])
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                #forward, track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs_array =  outputs.detach().cpu().clone().numpy()
                    for probs in outputs_array:
                        pred.append(probs[0])
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                several_labels = labels.data.cpu().numpy()
                for label in several_labels:
                    y_label.append(label)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            fpr, tpr, thresholds = metrics.roc_curve(y_label, pred, pos_label = 1)
            epoch_auc = 1 - metrics.auc(fpr, tpr)
            if epoch == 19:
                print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_auc))
            
            if phase == 'val' and epoch_auc > best_auc:
                best_acc = epoch_acc
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    print('saving model...')

    torch.save(model, entire_model_save_path)
    torch.save(model.state_dict(), state_save_path)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, check_point_save_path)
    #print('model saved to ' + state_save_path)
    return model


def test_model_with_subset(data_dir,subset, flipped_label_dict, models):
    mislabel_list = [] 
    num_classifiers = len(models)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=1) for x in ['train', 'val']} 
    '''
    print('-------------subset-------------')
    print(subset)
    print('-------------subset-------------')
    '''

    with torch.no_grad():
        for inputs, path in dataloaders['train']:
            error_counter = 0
            image_name = path[0].split('/')[-1] 
            if image_name not in subset:
                continue
            images = inputs.to(device)
            if image_name.startswith('n'):
                truelabel = torch.tensor([1]) 
            else:
                truelabel = torch.tensor([0]) 
            truelabel = truelabel.to(device)
            for j in range(num_classifiers):
                outputs = models[j](images)
                _, predicted = torch.max(outputs.data, 1)
                if predicted != truelabel:
                    error_counter += 1  
            if error_counter > num_classifiers / 2.0:
                mislabel_list.append(image_name)
    return tuple(mislabel_list)

def train_model_without_subset(model_name, data_dir, subset, flipped_label_dict):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_epochs = 20 
    #print('loading the model...')
    model = load_model(model_name)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    #print('start finetuning...')
    criterion = nn.CrossEntropyLoss()
    optimizer_fit = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_fit, step_size = 7, gamma = 0.1)
    model = train_model(model, model_name, criterion, optimizer_fit, exp_lr_scheduler, n_epochs, data_dir, subset, flipped_label_dict)
    return model


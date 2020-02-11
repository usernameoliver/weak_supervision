'''
@Auther: Degan Hao
@Date: 02/03/2020

'''

import torch
from torch.autograd import Variable
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
from influence_function import influence
import torch.nn.functional as F
import pickle
import sys

BASE_DIR = '/pylon5/ca5phjp/deh95/mi/'#Change the BASE_DIR to your working directory
#BASE_DIR = './'#Change the BASE_DIR to your working directory

def set_parameter_requires_grad(model, nth_frozen):
    count = 0
    for param in model.parameters():
        if count > nth_frozen:
            break
        count += 1
        param.requires_grad = False

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
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class ImageFolderWithPaths(datasets.ImageFolder):
    '''
    Custom dataset that includes image file paths by extending torchvision.datasets.ImageFolder
    '''
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple[0], path)
        return tuple_with_path

data_dir = BASE_DIR + 'training/data_breast_density/'
image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,shuffle=True, num_workers=1) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print('dataset_sizes =' + str(dataset_sizes)) 
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
n_epochs = 20 

verbose = False 

nth_layer = -1
flip_ratio = float(sys.argv[1]) 
f_train = open(BASE_DIR + '/training/label_dict/' + str(int(100 * flip_ratio)) + 'percent_flipped_label_train.pkl', "rb")
f_val = open(BASE_DIR + '/training/label_dict/' + str(int(100 * flip_ratio)) + 'percent_flipped_label_val.pkl', "rb")
flipped_label_dict_train = pickle.load(f_train)
flipped_label_dict_val = pickle.load(f_val)
flipped_label_dict = {**flipped_label_dict_train, **flipped_label_dict_val}

def train_model(model, criterion, optimizer, scheduler, num_epochs = n_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)
        count = 0
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_corrects_mislabels = 0
            pred = []
            pred_mislabels = []
            y_label = []
            y_label_mislabels = []
            running_auc = 0.0
            for inputs, path in dataloaders[phase]:
                inputs = inputs.to(device)
                image_name = path[0].split('/')[-1]
                labels = flipped_label_dict[image_name]
                labels = torch.tensor([labels]).to(device)
                optimizer.zero_grad()
                
                #forward, track history only in train
                set_grad = False
                if phase == 'train':
                    set_grad = True
                with torch.set_grad_enabled(set_grad):
                    outputs = model(inputs)
                    
                    outputs_softmax = F.softmax(outputs, dim = 1)
                    outputs_array =  outputs_softmax.detach().cpu().clone().numpy()
                    for probs in outputs_array:
                        pred.append(probs[0])
                    _, preds = torch.max(outputs_softmax, 1)
                    loss = criterion(outputs_softmax, labels)

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
            print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_auc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    print('saving model...')
    state_save_path = BASE_DIR + 'models/' + str(int(flip_ratio * 100)) + 'percent_flip_inceptionV3_influence_bc_state.pt'
    entire_model_save_path = BASE_DIR + 'models/' + str(int(flip_ratio * 100)) + 'percent_flip_inceptionV3_influence_bc_entire_model.pt'
    check_point_save_path = BASE_DIR + 'models/' + str(int(flip_ratio * 100)) + 'percent_flip_inceptionV3_influence_bc_checkpoint_model.pt'
    torch.save(model, entire_model_save_path)
    torch.save(model.state_dict(), state_save_path)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, check_point_save_path)
    print('model saved to ' + state_save_path)
    return model, loss

model = torchvision.models.inception_v3(pretrained=True) 
nth_frozen = 289 #217

model.aux_logits=False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

print('start training...')
criterion = nn.CrossEntropyLoss()
optimizer_fit = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_fit, step_size = 7, gamma = 0.1)

model, loss = train_model(model, criterion, optimizer_fit, exp_lr_scheduler, num_epochs = n_epochs)
print('flip ratio = ' + str(flip_ratio))
print('done')

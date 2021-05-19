from numpy.random import permutation
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import neptune.new as neptune
import numpy as np


def main():
    # Setting Up Neptune run 
    # Step 1: Initialize you project
    run = neptune.init(project='common/pytorch-integration', api_token='ANONYMOUS', source_files=['*.py'])

    # Experiment Config
    data_dir = 'data/CIFAR10'
    compressed_ds = './data/CIFAR10/cifar-10-python.tar.gz'
    data_tfms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    params = {
        'lr': 1e-2,
        'bs': 128,
        'input_sz': 32 * 32 * 3,
        'n_classes': 10,
        'epochs': 10,
        'model_filename': 'basemodel',
    }

    # Model & Dataset
    class BaseModel(nn.Module):
        def __init__(self, input_sz, hidden_dim, n_classes):
            super(BaseModel, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_sz, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, n_classes)
            )

        def forward(self, input):
            x = input.view(-1, 32* 32 * 3)
            return self.main(x)

    trainset = datasets.CIFAR10(data_dir, transform=data_tfms['train'], download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params['bs'], shuffle=True, num_workers=2)
    dataset_size = {'train': len(trainset)}

    # Instatiate model, crit & opt
    model = BaseModel(params['input_sz'], params['input_sz'], params['n_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['lr'])

    # Step 2: Log config & pararameters
    run['config/dataset/path'] = data_dir
    run['config/dataset/transforms'] = data_tfms
    run['config/dataset/size'] = dataset_size
    run['config/model'] = type(model).__name__
    run['config/criterion'] = type(criterion).__name__
    run['config/optimizer'] = type(optimizer).__name__
    run['config/params'] = params

    epoch_loss = 0.0
    epoch_acc = 0.0
    best_acc = 0.0

    # Step 3: Log losses & metrics 
    for epoch in range(params['epochs']):
        running_loss = 0.0
        running_corrects = 0
        
        for i, (x, y) in enumerate(trainloader, 0):

            optimizer.zero_grad()
            outputs = model.forward(x)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, y)
            acc = (torch.sum(preds == y.data)) / len(x)
            # Log batch loss
            run["logs/training/batch/loss"].log(value = loss)

            # Log batch accuracy
            run["logs/training/batch/acc"].log(value = acc)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == y.data)
            
        epoch_loss = running_loss/dataset_size['train']
        epoch_acc = running_corrects.double().item() / dataset_size['train']

        # Log epoch loss
        run[f"logs/training/epoch/loss"].log(value = epoch_loss, step = epoch + 1)

        # Log epoch accuracy
        run[f"logs/training/epoch/acc"].log(value = epoch_acc, step = epoch + 1)

        print(f'Epoch:{epoch+1}, Loss: {epoch_loss}, Acc: {epoch_acc}')
    
    # Step 5: Explore results in Neptune UI

    # Stop logging
    run.stop()

if __name__ == '__main__':
    main()

from numpy.random import permutation
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import neptune.new as neptune
import hashlib
import numpy as np

import os
os.environ['NEPTUNE_API_TOKEN']= 'ANONYMOUS'

# Setting Up Neptune run 

# Step 1: Initialize you project
run = neptune.init(project = 'common/pytorch-integration')

# Step 2: Create a namespace for all your experiment metadata
base_namespace = 'experiment'
ns_run = run[base_namespace]

# Helper function

def save_model(model, name ='model.txt'):
    print(f'Saving model arch as {name}.txt')
    with open(f'{name}_arch.txt', 'w') as f:  f.write(str(model))
    print(f'Saving model weights as {name}.pth')
    torch.save(model.state_dict(), f'./{name}.pth')

# Experiment Config
data_dir = 'data/CIFAR10'
compressed_ds = './data/CIFAR10/cifar-10-python.tar.gz'
sha = hashlib.sha1(compressed_ds.encode())
data_tfms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        ])
    }

parameters = {'lr': 1e-2,
              'bs': 128,
              'input_sz': 32 * 32 * 3,
              'n_classes': 10,
              'epochs': 2,
              'model_filename': 'basemodel',
              'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

trainset = datasets.CIFAR10(data_dir, transform=data_tfms['train'], 
                            download=True)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=parameters['bs'],
                                          shuffle=True, num_workers=2)

      
dataset_size = {'train': len(trainset), 'val': len(validset)}

# iN
model = BaseModel(parameters['input_sz'], parameters['input_sz'], parameters['n_classes']).to(parameters['device'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=parameters['lr'])

# Training Loop

# Step 3: Logging config & pararameters
ns_run['io_files/resource'] = (compressed_ds, sha.hexdigest())
ns_run['config/dataset/path'] = data_dir
ns_run['config/dataset/transforms'] = data_tfms
ns_run['config'] = parameters  
ns_run['config/dataset/size'] = dataset_size
ns_run['config/model'] = type(model).__name__
ns_run['config/criterion'] = type(criterion).__name__
ns_run['config/optimizer'] = type(optimizer).__name__

epoch_loss = 0.0
epoch_acc = 0.0
best_acc = 0.0

# Step 4: Log metrics and artifacts
for epoch in range(parameters['epochs']):
    running_loss = 0.0
    running_corrects = 0

    for i, (x, y) in enumerate(trainloader, 0):
        x, y = x.to(parameters['device']), y.to(parameters['device'])
        optimizer.zero_grad()
        outputs = model.forward(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        # Log batch loss
        ns_run["training/batch/loss"].log(value = loss)

        # Log batch accuracy
        ns_run["training/batch/acc"].log(value = acc)

        loss.backward()

        # logging grad_norm
        for p in list(filter(lambda p: p.grad is not None, model.parameters())): 
            ns_run['training/grad_norm'].log((p.grad.data.norm(2).item()))

        optimizer.step()
    

        running_loss += loss.item()
        running_corrects += torch.sum(preds == y.data)
        
    epoch_loss = running_loss/dataset_size['train']
    epoch_acc = running_corrects.double().item() / dataset_size['train']

    # Log epoch loss
    ns_run[f"training/epoch/loss"].log(value = epoch_loss, step = epoch + 1)

    # Log epoch accuracy
    ns_run[f"training/epoch/acc"].log(value = epoch_acc, step = epoch + 1)

    print(f'Epoch:{epoch+1}, Loss: {epoch_loss}, Acc: {epoch_acc}')
    if epoch_acc > best_acc:
        best_acc = epoch_acc

        # Saving model arch & weights
        save_model(model, parameters['model_filename'])
        print('Saving model -- Done!')

# Log model arch & weights
ns_run[f"io_files/artifacts/{parameters['model_filename']}_arch"].upload(f"./{parameters['model_filename']}_arch.txt")
ns_run[f"io_files/artifacts/{parameters['model_filename']}"].upload(f"./{parameters['model_filename']}.pth")

# Explore results in Neptune UI

# Stop logging
run.stop()
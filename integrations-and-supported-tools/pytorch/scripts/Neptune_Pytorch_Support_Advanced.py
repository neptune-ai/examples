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
# TODO: add sources
run = neptune.init(project = 'common/pytorch-integration')

# Helper functions
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
        ])
    }

hparams = {
    'lr': 1e-2,
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
                                          batch_size=hparams['bs'],
                                          shuffle=True, num_workers=2)
dataset_size = {'train': len(trainset), 'val': len(validset)}

# Instatiate model, crit & opt
model = BaseModel(hparams['input_sz'], hparams['input_sz'], hparams['n_classes']).to(hparams['device'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.hparams(), lr=hparams['lr'])

# Step 2: Log config & hyperpararameters
run['config/dataset/path'] = data_dir
run['config/dataset/transforms'] = data_tfms
run['config/dataset/size'] = dataset_size
run['config/model'] = type(model).__name__
run['config/criterion'] = type(criterion).__name__
run['config/optimizer'] = type(optimizer).__name__
run['config'] = hparams

epoch_loss = 0.0
epoch_acc = 0.0
best_acc = 0.0

# Step 3: Log losses and metrics 
for epoch in range(hparams['epochs']):
    running_loss = 0.0
    running_corrects = 0
    
    for i, (x, y) in enumerate(trainloader, 0):
        x, y = x.to(hparams['device']), y.to(hparams['device'])
        optimizer.zero_grad()
        outputs = model.forward(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        # Log batch loss
        run["training/batch/loss"].log(value = loss)

        # Log batch accuracy
        run["training/batch/acc"].log(value = acc)

        loss.backward()

        # logging grad_norm
        for p in list(filter(lambda p: p.grad is not None, model.hparams())): 
            run['training/grad_norm'].log((p.grad.data.norm(2).item()))

        optimizer.step()
    

        running_loss += loss.item()
        running_corrects += torch.sum(preds == y.data)
        
    epoch_loss = running_loss/dataset_size['train']
    epoch_acc = running_corrects.double().item() / dataset_size['train']

    # Log epoch loss
    run[f"training/epoch/loss"].log(value = epoch_loss, step = epoch + 1)

    # Log epoch accuracy
    run[f"training/epoch/acc"].log(value = epoch_acc, step = epoch + 1)

    print(f'Epoch:{epoch+1}, Loss: {epoch_loss}, Acc: {epoch_acc}')
    if epoch_acc > best_acc:
        best_acc = epoch_acc

        # Saving model arch & weights
        save_model(model, hparams['model_filename'])
        print('Saving model -- Done!')


# Step 4: Log model arch & weights -- > link to adding artifacts
run[f"io_files/artifacts/{hparams['model_filename']}_arch"].upload(f"./{hparams['model_filename']}_arch.txt")
run[f"io_files/artifacts/{hparams['model_filename']}"].upload(f"./{hparams['model_filename']}.pth")


# Step 5: Log images and predictions
# TODO

# Step 6: Explore results in Neptune UI

# Stop logging
run.stop()
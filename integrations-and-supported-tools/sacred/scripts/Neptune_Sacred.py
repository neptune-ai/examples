import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sacred import Experiment

import neptune.new as neptune
from neptune.new.integrations.sacred import NeptuneObserver

if torch.device("cuda:0"):
    torch.cuda.empty_cache()

# Step 1: Initialize Neptune and create new Neptune Run
neptune_run = neptune.init(
    project='common/sacred-integration', 
    api_token = 'ANONYMOUS',
    tags = 'notebook'
)

# Step 2: Add NeptuneObserver() to your sacred experiment's observers
ex = Experiment('image_classification', interactive=True)
ex.observers.append(NeptuneObserver(run=neptune_run))


class BaseModel(nn.Module):
    def __init__(self, input_sz = 32 * 32 * 3, n_classes = 10):
        super(BaseModel, self).__init__()
        self.lin = nn.Linear(input_sz, n_classes)


    def forward(self, input):
        x = input.view(-1, 32 * 32 * 3)
        return self.lin(x)

model = BaseModel()

# Log hyperparameters
@ex.config
def cfg():
    data_dir = 'data/CIFAR10'
    data_tfms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    lr = 1e-2
    bs = 128
    n_classes = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Log loss and metrics
@ex.main
def run(data_dir, data_tfms, input_sz, n_classes, lr, bs, device, _run):
    
    trainset = datasets.CIFAR10(data_dir, transform=data_tfms['train'], 
                                download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True, num_workers=2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)  
    for i, (x, y) in enumerate(trainloader, 0):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model.forward(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        # Log loss
        ex.log_scalar("training/batch/loss", loss)
        # Log accuracy
        ex.log_scalar("training/batch/acc", acc)

        loss.backward()
        optimizer.step()
    
    return {'final_loss': loss.item(), 'final_acc': acc.cpu().item()}

# Step 3: Run you experiment and explore metadata in Neptune UI
ex.run()

# Stop run  
neptune_run.stop()
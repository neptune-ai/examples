from numpy.random import permutation
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import neptune.new as neptune
import numpy as np

def main():
    # Setting Up Neptune run 
    # Step 1: Initialize you project
    run = neptune.init(project='common/pytorch-integration', api_token='ANONYMOUS', source_files=['*.py'])

    # Helper functions
    def save_model(model, name ='model.txt'):
        print(f'Saving model arch as {name}.txt')
        with open(f'{name}_arch.txt', 'w') as f:  f.write(str(model))
        print(f'Saving model weights as {name}.pth')
        torch.save(model.state_dict(), f'./{name}.pth')

    # Experiment Config
    data_dir = 'data/CIFAR10'
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

    params = {
        'lr': 1e-2,
        'bs': 128,
        'input_sz': 32 * 32 * 3,
        'n_classes': 10,
        'epochs': 1,
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
                                            batch_size=params['bs'],
                                            shuffle=True, num_workers=2)

    validset = datasets.CIFAR10(data_dir, train=False,
                            transform=data_tfms['train'],
                            download=True)
    validloader = torch.utils.data.DataLoader(validset, 
                                          batch_size=params['bs'], 
                                          num_workers=2)
    dataset_size = {'train': len(trainset), 'val': len(validset)}

    # Instatiate model, crit & opt
    model = BaseModel(params['input_sz'], params['input_sz'], params['n_classes']).to(params['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['lr'])

    # Step 2: Log config & hyperpararameters
    run['config/dataset/path'] = data_dir
    run['config/dataset/transforms'] = data_tfms
    run['config/dataset/size'] = dataset_size
    run['config/model'] = type(model).__name__
    run['config/criterion'] = type(criterion).__name__
    run['config/optimizer'] = type(optimizer).__name__
    run['config/hyperparameters'] = params

    epoch_loss = 0.0
    epoch_acc = 0.0
    best_acc = 0.0

    # Step 3: Log losses and metrics 
    for epoch in range(params['epochs']):
        running_loss = 0.0
        running_corrects = 0
        
        for i, (x, y) in enumerate(trainloader, 0):
            x, y = x.to(params['device']), y.to(params['device'])
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
            save_model(model, params['model_filename'])
            print('Saving model -- Done!')


    # Step 4: Log model arch & weights -- > link to adding artifacts
    run[f"io_files/artifacts/{params['model_filename']}_arch"].upload(f"./{params['model_filename']}_arch.txt")
    run[f"io_files/artifacts/{params['model_filename']}"].upload(f"./{params['model_filename']}.pth")


    # Step 5: Log Images & Predictions
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog','horse','ship','truck']
    dataiter = iter(validloader)
    images, labels = dataiter.next()
    images_arr = []
    labels_arr = []
    pred_arr = []
    model.eval()
    # moving model to cpu for inference 
    if torch.cuda.is_available(): model.to("cpu")
    # iterating on the dataset to predict the output
    for i in range(0,10):
        images_arr.append(images[i].unsqueeze(0))
        labels_arr.append(labels[i].item())
        ps = torch.exp(model(images_arr[i]))
        ps = ps.data.numpy().squeeze()
        pred_arr.append(np.argmax(ps))
    # plotting the results
    fig = plt.figure(figsize=(25,4))
    for i in range(10):
        ax = fig.add_subplot(2, 20/2, i+1, xticks=[], yticks=[])
        ax.imshow(images_arr[i].squeeze().permute(2,1,0).numpy().clip(0,1))
        ax.set_title("{} ({})".format(classes[pred_arr[i]], classes[labels_arr[i]]),
                    color=("green" if pred_arr[i]==labels_arr[i] else "red"))

    fig.savefig('predictions.jpg')

    # Log image with predictions
    run['io_files/artifacts/predictions'].upload('predictions.jpg')

    # Step 6: Explore results in Neptune UI

    # Stop logging
    run.stop()


if __name__ == "__main__":
    main()
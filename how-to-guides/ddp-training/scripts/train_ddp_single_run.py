import os

import neptune
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def create_data_loader_cifar10(rank, batch_size):
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=trainset, rank=rank)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=testset, rank=rank)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
    )
    return trainloader, testloader


def train(net, trainloader, run, rank, params):
    if rank == 0:
        # (Neptune) Log params
        run["parameters"] = params

    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=params["lr"], momentum=params["momentum"])
    num_of_batches = len(trainloader)
    for epoch in range(params["epochs"]):
        trainloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            images, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if rank == 0:
            epoch_loss = running_loss / num_of_batches
            # (Neptune) Log metrics
            run["metrics/train/loss"].append(epoch_loss)
            print(f'[Epoch {epoch + 1}/{params["epochs"]}] loss: {epoch_loss:.3f}')

    print("Finished Training")


def test(net, testloader, run, rank):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if rank == 0:
        acc = 100 * correct // total
        # (Neptune) Log metrics
        run["metrics/valid/acc"] = acc
        print(f"Accuracy of the network on the 10000 test images: {acc} %")


def init_distributed():
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)

    # This will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # Synchronizes all the threads to reach this point before moving on
    dist.barrier()


if __name__ == "__main__":
    init_distributed()

    rank = rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    params = {"batch_size": 256, "lr": 0.001, "epochs": 2, "momentum": 0.9}

    trainloader, testloader = create_data_loader_cifar10(rank=rank, batch_size=params["batch_size"])

    net = torchvision.models.resnet50(weights=None).cuda()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    # To correctly monitor each GPU usage
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # No. of GPUs installed

    # (Neptune) Create new run on the main process only
    run = (
        neptune.init_run(project="common/showroom", api_token=neptune.ANONYMOUS_API_TOKEN)
        if rank == 0
        else None
    )

    # Train model
    train(net, trainloader, run, rank, params)

    # Test model
    test(net, testloader, run, rank)

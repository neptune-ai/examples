import hashlib
import os
import time

import neptune.new as neptune
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
        num_workers=14,
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
        num_workers=14,
    )
    return trainloader, testloader


def train(net, trainloader, run, rank, params):
    if rank == 0:
        # Log params
        run["parameters"] = params

    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 2
    num_of_batches = len(trainloader)
    for epoch in range(epochs):
        trainloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            images, labels = inputs.to(f"cuda:{rank}"), labels.to(f"cuda:{rank}")
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Gather loss value from all processes on main process for logging
            dist.reduce(tensor=loss, dst=0)
            dist.barrier()  # synchronizes all the threads

            if rank == 0:
                running_loss += loss.item() / dist.get_world_size()

        if rank == 0:
            epoch_loss = running_loss / num_of_batches
            # Log metrics
            run["metrics/train/loss"].log(epoch_loss)
            print(f"[Epoch {epoch + 1}/{epochs}] loss: {epoch_loss:.3f}")

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

            # Gather labels and predicted tensors from all processes on main process for logging
            dist.reduce(tensor=labels, dst=0)
            dist.barrier()  # synchronizes all the threads

            dist.reduce(tensor=predicted, dst=0)
            dist.barrier()  # synchronizes all the threads

            if rank == 0:
                correct += (predicted == labels).sum().item()

    if rank == 0:
        acc = 100 * correct // total
        # Log metrics
        run["metrics/valid/acc"] = acc
        print(f"Accuracy of the network on the 10000 test images: {acc} %")


def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
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

    # Automatically create and broadcast `custom_run_id` to all processes
    if rank == 0:
        custom_run_id = [hashlib.md5(str(time.time()).encode()).hexdigest()]
        monitoring_namespace = "monitoring"
    else:
        custom_run_id = [None]
        monitoring_namespace = f"monitoring/{rank}"

    dist.broadcast_object_list(custom_run_id, src=0)
    custom_run_id = custom_run_id[0]

    # Creates multiple run instances
    # But all instances log metadata to the same run
    # by passing the `custom_run_id` argument
    run = neptune.init_run(
        project="common/showroom",
        api_token=neptune.ANONYMOUS_API_TOKEN,
        monitoring_namespace=monitoring_namespace,
        custom_run_id=custom_run_id,
    )

    # Train model
    train(net, trainloader, run, rank, params)

    # Test model
    test(net, testloader, run, rank)
import neptune
import torch
import torch.nn.functional as F
from ignite.contrib.handlers.neptune_logger import (
    GradsScalarHandler,
    NeptuneLogger,
    NeptuneSaver,
    WeightsScalarHandler,
    global_step_from_engine,
)
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# Define hyper-parameters
params = {
    "train_batch_size": 64,
    "val_batch_size": 64,
    "epochs": 10,
    "lr": 0.1,
    "momentum": 0.1,
}


# Create model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


model = Net()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # Move model before creating optimizer


# Define DataLoader()
def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True),
        batch_size=train_batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False),
        batch_size=val_batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


train_loader, val_loader = get_data_loaders(params["train_batch_size"], params["val_batch_size"])

# Create optimizer, trainer, and logger
optimizer = SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"])
criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

# (Neptune) Create NeptuneLogger()
neptune_logger = NeptuneLogger(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/pytorch-ignite-integration",
)

# (Neptune) Attach logger to the trainer
trainer.logger = setup_logger("Trainer")

neptune_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batchloss": loss},
)

# Create evaluators
metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}

train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


@trainer.on(Events.EPOCH_COMPLETED)
def compute_metrics(engine):
    train_evaluator.run(train_loader)
    validation_evaluator.run(val_loader)


# (Neptune) Attach logger to training and validation evaluators
train_evaluator.logger = setup_logger("Train Evaluator")

neptune_logger.attach_output_handler(
    train_evaluator,
    event_name=Events.EPOCH_COMPLETED,  # logging at the end of each epoch
    tag="training",
    metric_names="all",
    global_step_transform=global_step_from_engine(
        trainer
    ),  # takes the epoch of the trainer instead of train_evaluator
)

validation_evaluator.logger = setup_logger("Validation Evaluator")

neptune_logger.attach_output_handler(
    validation_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="validation",
    metric_names="all",
    global_step_transform=global_step_from_engine(
        trainer
    ),  # takes the epoch of the trainer instead of train_evaluator
)

# (Neptune) Log optimizer parameters
neptune_logger.attach_opt_params_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    optimizer=optimizer,
)

# (Neptune) Log model's normalized weights and gradients after each iteration
neptune_logger.attach(
    trainer,
    log_handler=WeightsScalarHandler(model, reduction=torch.norm),
    event_name=Events.ITERATION_COMPLETED(every=100),
)

neptune_logger.attach(
    trainer,
    log_handler=GradsScalarHandler(model, reduction=torch.norm),
    event_name=Events.ITERATION_COMPLETED(every=100),
)


# (Neptune) Save model checkpoints
# Note: `NeptuneSaver` currently does not work on Windows
def score_function(engine):
    return engine.state.metrics["accuracy"]


to_save = {"model": model}

handler = Checkpoint(
    to_save=to_save,
    save_handler=NeptuneSaver(neptune_logger),
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="validation_accuracy",
    global_step_transform=global_step_from_engine(trainer),
)

# validation_evaluator.add_event_handler(Events.COMPLETED, handler) # Uncomment to save model checkpoints on MacOS/Linux

# Run trainer
trainer.run(train_loader, max_epochs=params["epochs"])

# (Neptune) Log hyper-parameters
neptune_logger.experiment["params"] = params

# (Neptune) Upload trained model
torch.save(model.state_dict(), "model.pth")
neptune_logger.experiment["trained_model"].upload("model.pth")

# (Neptune) Stop logging
neptune_logger.close()

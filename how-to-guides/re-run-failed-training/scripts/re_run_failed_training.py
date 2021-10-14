import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms

"""## Step 1: Get Run ID
You will get the Run ID of the failed run **programmatically**.
"""

# Fetch project
project = neptune.get_project(name='common/showroom', api_token='ANONYMOUS')

# Fetch only inactive runs
runs_table_df = project.fetch_runs_table(state="idle", tag=['showcase-run']).to_pandas()

# Sort runs by failed 
runs_table_df = runs_table_df.sort_values(by='sys/failed', ascending=True)

# Extract the last failed run's id
failed_run_id = runs_table_df[runs_table_df['sys/failed']==True]['sys/id'].values[0]

print('Failed_run_id = ', failed_run_id)

"""## Step 2: Resume run
Use the neptune.init() method to:
* Re-open a run using the ID you got from the previous step 
* Re-open it in the `read-only` mode

You use the `read-only` mode so the metadata previously logged to the run is not accidentally changed. Also, you can re-open a run as many times as needed.

"""

failed_run = neptune.init(
    project="common/showroom",
    api_token="ANONYMOUS",
    mode="read-only",
    run=failed_run_id
)

"""## Step 3: Fetching and downloading data from Neptune

Fetch metadata(dataset and hyperparameters) needed to re-run the training. Precisely, you will download the hyperparameters used in the failed run to instantiate a model with the same configuration and then you will download the dataset path to get the same dataset too.

To do that:

### Use the  [.download()](https://docs.neptune.ai/api-reference/field-types#.download-3) method to retrieve the [dataset artifact](https://docs.neptune.ai/api-reference/field-types#artifact) to your local disk:
"""

data_dir = 'data'

failed_run['artifacts/dataset'].download(destination=data_dir)

"""###Use the [.fetch()](https://docs.neptune.ai/api-reference/field-types#.fetch-1) method to retrieve hyperparameters:"""

# fetching non-file values 
failed_run_params = failed_run['config/hyperparameters'].fetch()

"""## Step 4: Create a new run
Create a new Neptune run that will be used to log metadata in the re-run session.
"""

new_run = neptune.init(
    project="common/showroom",
    tags=['re-run', 'successful training'],
    api_token="ANONYMOUS"
)

"""## Step 5: Log new training metadata
Now you can continue working and logging metadata to a brand new Run.
You can log metadata using the [Neptune API Client](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display).

### Log copy of dataset artifact from failed_run to new run
"""

new_run["artifacts/dataset"].assign(failed_run["artifacts/dataset"].fetch())

"""### Log Hyperparameters from failed run to new run"""

new_run["config/hyperparameters"] = failed_run_params

"""### Load Dataset and Model

Dataset
"""

data_tfms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

trainset = datasets.CIFAR10(data_dir+'/CIFAR10', transform=data_tfms["train"], download=False)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=failed_run_params["bs"], shuffle=True, num_workers=2
)

"""Model"""

class BaseModel(nn.Module):
    def __init__(self, input_sz, hidden_dim, n_classes):
        super(BaseModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_sz, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, input):
        x = input.view(-1, 32 * 32 * 3)
        return self.main(x)

model = BaseModel(
    failed_run_params["input_sz"], failed_run_params["input_sz"], failed_run_params["n_classes"]
).to(failed_run_params["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=failed_run_params["lr"])

"""### Log losses and metrics"""

for i, (x, y) in enumerate(trainloader, 0):
    x, y = x.to(failed_run_params["device"]), y.to(failed_run_params["device"])
    optimizer.zero_grad()
    outputs = model.forward(x)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, y)
    acc = (torch.sum(preds == y.data)) / len(x)

    new_run["training/batch/loss"].log(loss)

    new_run["training/batch/acc"].log(acc)

    loss.backward()
    optimizer.step()

"""# Stop run

<font color=red>**Warning:**</font><br>
Once you are done logging, you should stop tracking the run using the `stop()` method.
This is needed only while logging from a notebook environment. While logging through a script, Neptune automatically stops tracking once the script has completed execution.
"""

failed_run.stop()
new_run.stop()
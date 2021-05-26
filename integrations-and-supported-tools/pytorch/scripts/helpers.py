import torch
import numpy as np
import matplotlib.pyplot as plt
from neptune.new.types import File
import torch.nn.functional as F

# Helper functions
def save_model(model, name ="model.txt"):
    print(f"Saving model arch as {name}.txt")
    with open(f"{name}_arch.txt", "w") as f:  f.write(str(model))
    print(f"Saving model weights as {name}.pth")
    torch.save(model.state_dict(), f"./{name}.pth")
    
def get_obj_name(obj):
    return type(obj).__name__

def save_image_predictions(model, validloader, run, n_samples = 50):
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog","horse","ship","truck"]
    dataiter = iter(validloader)
    images, labels = dataiter.next()
    model.eval()

    # moving model to cpu for inference 
    if torch.cuda.is_available(): model.to("cpu")

    # predict batch of n_samples
    img = images[:n_samples]
    probs = F.softmax(model(img),dim=1)
    probs = probs.data.numpy()

    # Decode probs and Log images
    for i, ps in enumerate(probs):
        pred = classes[np.argmax(ps)]
        gt = classes[labels[i]]
        description = "\n".join(
            ["class {}: {}%".format(classes[n], round(p*100, 2)) for n, p in enumerate(ps)]
        )
        # Log Series of Tensors as Image and Predictions. For more see ->
        # https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#pytorch-tensor
        # and to understand how to upload multiple files see
        # https://docs.neptune.ai/api-reference/field-types#fileseries
        run.log(
            File.as_image(img[i].squeeze().permute(2,1,0).clip(0,1)), 
            name=f'{i}_{pred}_{gt}', 
            description=description
        )
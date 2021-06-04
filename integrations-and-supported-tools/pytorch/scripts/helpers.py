import torch
import numpy as np
import torch.nn.functional as F

# Helper functions
def save_model(model, name ="model.txt"):
    print(f"Saving model arch as {name}.txt")
    with open(f"{name}_arch.txt", "w") as f:  f.write(str(model))
    print(f"Saving model weights as {name}.pth")
    torch.save(model.state_dict(), f"./{name}.pth")
    
def get_obj_name(obj):
    return type(obj).__name__

def get_predictions(model, validloader, n_samples = 50):
    dataiter = iter(validloader)
    images, labels = dataiter.next()
    model.eval()

    # moving model to cpu for inference 
    if torch.cuda.is_available(): model.to("cpu")

    # predict batch of n_samples
    img = images[:n_samples]
    probs = F.softmax(model(img),dim=1)


    return probs, img, labels


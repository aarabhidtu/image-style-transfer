import torch 
import torch.optim as optim
from torchvision.models import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# tensors are multi-dimensional arrays with a uniform type 
# they are a specialized data structure that are very similar to arrays and matrices
# tensors can run on GPUs or other hardware accelerators

# transforms are common image transformations
# they can be chained together using Compose
# all transforms accept PIL images/tensor images and return PIL images

# torchvision is a library that provides tools for working with images and videos
# PIL is the Python Imaging Library, which provides tools for working with images


# loading and processing images
def load_img(image_path, max_size=400):
    img=Image.open(image_path).convert('RGB')
    size= min(max(img.size), max_size)
    transform=transforms.Compose([          # Compose is used to chain multiple image transformations together. Each transformation is applied sequentially to the image.
        transforms.Resize((size, size)),    # this transformation resizes the input image to a new size
        transforms.ToTensor(),            # this transformation converts a PIL Image or numpy.ndarray to a tensor
    ])
    img=transform(img).unsqueeze(0)        # transform(image) applies all the transformations defined above to the image
                                           # This results in a transformed tensor of shape [C,H,W] C-> number of channels, H->height, W->width
                                           # unsqueeze(0) adds an extra dimension at the beginning of the tensor ( for the batch B=1). Indicating the batch size is 1
    return img

# displaying the images
def display(tensor, title=None):
    # we perform 2 operations on the tensor: clone() and detach()
    # clone() -> creates a deep copy of the tensor i.e a new tensor is created with the same data as the original tensor but it doesn't share the same memory
    # hence changes made on the new tensor will not affect the original tensor
    # basically gradients represent the partial derivatives of a model's parameters with respect to the loss function
    # these are crutial as they guide how to update the model's parameters in order to minimize the loss function  
    # tensors are involved in orperations such as matrix multiplication in a neural network will have gradients associated with them. These
    # gradients are used to update the parameters during the training process
    # detach() -> essentially says that any subsequent operations performed on the detached tensor will not have gradients computed for them.
    image=tensor.clone().detach() 
    image=image.squeeze(0)        # reduces the tensor back to the original shape by removing one dimension
                                  # we usually unsqueeze when the model expects the input parameter to include batches
    image=transforms.ToPILImage()(image) 
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()


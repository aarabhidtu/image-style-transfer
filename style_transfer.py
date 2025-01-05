import torch 
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import ssl

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

# extracting features 
# model-> represents the neural network
# _modules-> is an internal attribute that stores all the layers defined in the module
# model._modules.items() -> provides an iterator over the layers of the model
# each item is a key-value pair:
    # key-> name of the layer (string)
    # value-> the layer object itself
# the loop iterates over all the layers of the model, applying each layer to the input data
# the output of each layer is stored in the features dictionary
# the output x of the layer becomes the input to the next layer
# layers-> list of layers whose outputs we want to extract

def extract_features(image, model, layers):
    features={}
    x=image
    for name, layer in model._modules.items():
        x=layer(x)         # applying the layer to the input image/ the output of the previous layer
        if name in layers: # storing the ouput for only the layers we want
            features[name]=x
    return features        # dictionary containing the output tensor values of the layers we want

# gram matrix computation
# used to capture style from paintings. why and how?
# how? -> it is computed by taking the dot product of the image matrix and it's transpose
# why? -> first, different feature maps represent the different features of an image but these features are independent from each other
#         so, it can capture objects and shapes but cannot capture the style of an image
#         second, style refers to the texture, brush stroke, colours etc of an image. The occurance of these are dependent on each other
#         so, if we can measure the degree of correlation between these matrices, we can represent the style of an image
# hence, the correlation of feature maps can be calculated using dot products 
# to do this, we convert the feature maps into a vector and multiple this vector with it's transpose, obtaining the style of an image 
# the gram matrix is a square matrix that represents the correlation between the feature maps

def gram_matrix(tensor):
    _,d,h,w=tensor.size() # _ -> batch size
    tensor=tensor.view(d,h*w) # reshaping the tensor into a 2D tensor of shape (d, h*w). d-> number of rows. h*w-> total number of pixels becomes the number of columns
    # example: before reshaping- (1,3,4,4). after reshaping- (3,16)
    return torch.mm(tensor, tensor.t()) # computing the dot product of the tensor and it's transpose

# computing the loss
# typically the loss calculates the difference between the target and the output
def compute_loss(content_features, style_features, target_features, content_weight=1e4, style_weight=1e-2):
    # content loss measures how much the content of the target image deviates from the content of the original image
    # we square the difference to ensure that all 
    # mean avergaes the squared differences to provide a single scalar value
    content_loss=torch.mean((target_features['conv4_2']-content_features['conv4_2'])**2) # the difference tells us how far apart the two feature maps are from each other
    style_loss=0
    for layer in style_features:
        target_gram=gram_matrix(target_features[layer]) # calculates the gram matrix of the target image
        style_gram= gram_matrix(style_features[layer]) # calculates the gram matrix of the style image
        style_loss+=torch.mean((target_gram-style_gram)**2) # calculates the loss between the gram matrices of the target and style images (style loss)
    return content_weight*content_loss+style_weight*style_loss

# the main function 
def style_transfer(content_path, style_path, output_path, epochs=300):
    # load the images
    content_img=load_img(content_path)
    style_img=load_img(style_path)

    # display the input images
    display(content_img, title='Content Image')
    display(style_img, title='Style Image')

    # load the pre-trained VGG19 model
    vgg=models.vgg19(pretrained=True).features.eval()

    # defining the content and style layers 
    content_layers=['conv4_2']
    style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    # extracting the feature 
    content_features=extract_features(content_img, vgg, content_layers)
    style_features=extract_features(style_img, vgg, style_layers)

    # initializing the target image
    target_img=content_img.clone().requires_grad_(True)

    # defining the optimizer
    # lr controls the step size for updates to the target image
    # gradients with respect to the target img are calculated using backpropogation
    optimizer=optim.Adam([target_img], lr=0.03)

    # training loop
    print('starting the style transfer...')
    for i in range(epochs):
        target_features= extract_features(target_img, vgg, content_layers+style_layers)
        loss=compute_loss(content_features, style_features, target_features)
        optimizer.zero_grad() # resets the gradients to zero
        loss.backward()       # it computes the gradients of the loss with respect to the parameters of the model using packpropogation
        optimizer.step()      # updates the parameters of the model using the gradients computed during the backward pass
        if i%50==0:
            print(f'Epoch {i}: Loss: {loss.item()}')
    
    # display the output image
    output_image=target_img.clone().detach()
    output_image=output_image.squeeze(0)
    output_image=transforms.ToPILImage()(output_image)
    output_image.save(output_path)
    print(f'stylized image saved at {output_path}')

    display(target_img, title='stylized image')

# paths 
content_path='images/content.jpg'
style_path='images/style.jpg'
output_path='output/output.jpg'
ssl._create_default_https_context = ssl._create_unverified_context

# style transfer
style_transfer(content_path, style_path, output_path)


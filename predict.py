#Import Statements
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
# Creating a parser for input arguments
parser = argparse.ArgumentParser()
parser.add_argument('imgPath', action = 'store',
                    type =str, default = 'flowers/test/10/image_07090.jpg', help="The image to predict the probabilities and classes")
parser.add_argument('filePath', action = 'store',
                    type =str, default = 'myCheckpoint.pth', help="A checkpoint to load the trained model.")
parser.add_argument('--topK', dest = 'topK', action = 'store',
                    type =int, default = 3, help="The top K Classes to be given.")
parser.add_argument('--gpu', dest = 'swGpu', action = 'store_true',
                    default = False, help="The device to be used. Default is CPU. Care must be taken to use the same device the model was trained on.")
parser.add_argument('--catName', dest = 'catName', action = 'store',
                    default = 'cat_to_name.json', help="A json file containing the mapping of catNames to class")


args = parser.parse_args()
imgPath = args.imgPath
filePath = args.filePath
topK = args.topK
swGpu = args.swGpu
catName = args.catName
# Data Directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

with open(catName, 'r') as f:
    cat_to_name = json.load(f)
# A model dictionay to call the Architecture of the model
modelDict = {'vgg13': models.vgg13,
             'vgg16': models.vgg16,
             'vgg19': models.vgg19,
             'densenet121': models.densenet121,
             'densenet169': models.densenet169,
             'densenet201': models.densenet201,}

#A function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    modArch = checkpoint['modArch']
    imgModel = modelDict[modArch](pretrained = True)
    imgModel.classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(checkpoint['inSize'], checkpoint['hidSize'])),
                                ('re1', nn.ReLU()),
                                ('do1', nn.Dropout()),
                                ('out', nn.Linear(checkpoint['hidSize'], checkpoint['outSize'])),
                                ('sof', nn.LogSoftmax(dim = 1))
                                ]))
    imgModel.load_state_dict(checkpoint['modelState'])
    imgModel.class_to_idx = checkpoint['classtoidx']
    devType = checkpoint['device']
    device = torch.device(devType)
    return imgModel, device

newModel, device = load_checkpoint(filePath)

# Function to process an image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Assigning the image to a variable
    im = image

    # Resizing the image using resize
    im = im.resize((256, 256))

    # Cropping the image with rectangular box coordinates, Coordinates in pixel images start at (0,0) upper left
    # and increase as you go down in both the x and y direction
    i2 = im.crop((16,16,240,240))

    #Converting the object to a Numpy array
    npIm = np.array(i2)

    #Scaling the image
    npIm = npIm/255
    npIm[:,:,0] = (npIm[:,:,0] - 0.485)/0.229
    npIm[:,:,1] = (npIm[:,:,1] - 0.456)/0.224
    npIm[:,:,2] = (npIm[:,:,2] - 0.406)/0.225

    #Changing the coordinates
    npIm = npIm.transpose(2,0,1)

    return npIm
# End of the process_image function

# Function to predict the image
def predict(image_path, model, topK= topK):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(Image.open(image_path))
    image = torch.from_numpy(image).type(torch.FloatTensor)
    model.eval()
    model.to(device)
    image = image.to(device)
    image = image.unsqueeze_(0)
    with torch.no_grad():
        output = newModel.forward(image)
    ps = torch.exp(output)
    probs= list(np.array(ps.topk(topK, dim=1)[0])[0])
    indexList = list(np.array(ps.topk(topK, dim=1)[1])[0])
    classes = []
    for item  in newModel.class_to_idx.items():
        if item[1] in indexList:
            classes.append(item[0])
    catNames = []
    for i  in classes:
        catNames.append(cat_to_name[i])
    return probs, catNames
# End of function
# Predicting using the predict function
probs , catNames = predict(imgPath, newModel, topK)
print(catNames)
print(probs)

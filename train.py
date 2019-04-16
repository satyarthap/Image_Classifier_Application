#import arguments here
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Creating a parser for input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataDir', action = 'store',
                    type =str, default = 'flowers', help="The directory to train the model on.")
parser.add_argument('--saveDir', dest = 'saveDir', action = 'store',
                    type =str, default = '/home/workspace/paind-project', help="The directory to save the model")
parser.add_argument('--arch', dest = 'modArch', action = 'store',
                    type =str, default = 'densenet201', choices = ['vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201'],
                    help="The architecture to be used in the model.")
parser.add_argument('--learnRate', dest = 'lrRate', action = 'store',
                    type =float, default = 0.001, help="Learning rate of the model.")
parser.add_argument('--hidUnits', dest = 'hidSize', action = 'store',
                    type =int, default = 512, help="The directory to train the model on.")
parser.add_argument('--epochs', dest = 'epochs', action = 'store',
                    type =int, default = 3, help="The directory to train the model on.")
parser.add_argument('--gpu', dest = 'swGpu', action = 'store_true',
                    default = True,
                    help="The device to train the model on. GPU is defaulted since such a huge model will take forever on a cpu")

args = parser.parse_args()
dataDir = args.dataDir
saveDir = args.saveDir
modArch = args.modArch
lrRate = args.lrRate
hidSize = args.hidSize
epochs = args.epochs
swGpu = args.swGpu

# Data Directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Data Transforms for the data
trainTransform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225] )])

validTransform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225] )])

testTransform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225] )])
# Loading the datasets
trainDataset = datasets.ImageFolder(train_dir, transform= trainTransform)

testDataset = datasets.ImageFolder(test_dir, transform=testTransform)

validDataset = datasets.ImageFolder(valid_dir, transform=validTransform)

#Loading the DataLoaders
trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size= 64, shuffle= True)

testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size= 32)

validDataLoader = torch.utils.data.DataLoader(validDataset, batch_size= 32)

# Building the model
imgModel = models.densenet201(pretrained= True)

for param in imgModel.parameters():
    param.requires_grad = False

# A model dictionay to call the Architecture of the model
modelDict = {'vgg13': models.vgg13,
             'vgg16': models.vgg16,
             'vgg19': models.vgg19,
             'densenet121': models.densenet121,
             'densenet169': models.densenet169,
             'densenet201': models.densenet201,}
# An input feature Dictionary that will bel useful in creating the classifier
inSizeDict = {'vgg13': 25088,
              'vgg16': 25088,
              'vgg19': 25088,
              'densenet121': 1024,
              'densenet169': 1664,
              'densenet201': 1920,
            }

#Loading the pretrained model and assigning it to our Model variable
imgModel = modelDict[modArch](pretrained = True)
#Freezing the parameters so we do not train the pretrained model
for param in imgModel.parameters():
    param.requires_grad = False
# We have already defined inSize and hidSize. Now we will be using 102 as outsize since we have 102 categories
outSize = 102
inSize = inSizeDict[modArch]
imgClassifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(inSize, hidSize)),
                                ('re1', nn.ReLU()),
                                ('do1', nn.Dropout(0.2)),
                                ('out', nn.Linear(hidSize, outSize)),
                                ('sof', nn.LogSoftmax(dim = 1))
]))

imgModel.classifier = imgClassifier

# Defining the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(imgModel.classifier.parameters(), lr = lrRate)

# Device to train the model on. GPU or CPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and swGpu) else "cpu")
devType = device.type
# A function to implement a validation pass on a given DataLoader
def validation(model, dataLoader, criterion):
    '''
    A function that validates a given validation data loader and returns loss and accuracy.

    Inputs : model , dataloader , criterion

    Outputs : loss, accuracy
    '''
    validLoss = 0
    accuracy = 0
    for images, labels in dataLoader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        validLoss += criterion(output, labels).item()

        ps = torch.exp(output)
        accuracy += (labels == ps.max(dim=1)[1]).type(torch.FloatTensor).mean()

    # Calculating the total Accuracy. accuracy after for loop is sum of accuracy of all batches in dataLoader.
    # Final Accuracy is accuracy/len(dataLoader)
    validLoss = validLoss/len(dataLoader)
    accuracy = accuracy/len(dataLoader)

    return validLoss, accuracy

#Training begins
# epochs can be inputed into the model. Default is 9.
print_every = 40
steps = 0
imgModel.to(device)
print('This will take some time..Grab a drink..')
for e in range(epochs):
    imgModel.train()
    runLoss = 0
    for images, labels in trainDataLoader:
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        #Forward pass, Backward pass and update the weights
        output = imgModel.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        runLoss += loss.item()

        if steps % print_every ==0:
            # Model is evaluation mode
            imgModel.eval()
            # Switching off gradients to save memory
            with torch.no_grad():
                validLoss, accuracy = validation(imgModel, validDataLoader, criterion)


            print('Epoch {}/{}'.format(e+1, epochs),
                 ('Train Loss: {:.4f}'.format(runLoss/print_every)),
                 ('Valid Loss: {:.4f}'.format(validLoss)),
                 ('Valid Accuracy:{:.4f}'.format(accuracy)))

            runLoss = 0

            imgModel.train()

# Training Ends

# Model checkpoint is saved in the save directory.

checkpoint ={'inSize': inSize,
              'hidSize':hidSize,
              'outSize': outSize,
              'modArch': modArch,
              'device': devType,
              'classtoidx': trainDataset.class_to_idx,
              'modelState':imgModel.state_dict(),
              'optimState': optimizer.state_dict(),
              'classifier': imgClassifier,
              'epoch': epochs}

torch.save(checkpoint, saveDir + '/myCheckpoint.pth')

print('Model saved to {} directory as {} '.format(saveDir, 'myCheckpoint.pth'))

# Image Classifier 
## Deep Learning
CLI application to classify 102 different species of flowers. 

This is a command line application that can be trained on any set of labelled images. Here the network is trained on 102 different flowers categories. This image classifier can predict an image of a flower as one of the 102 different species. 

## Prerequisites
The Code is written in Python 3.6.5 . If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. 

To install pip run in the command Line
```
python -m ensurepip -- default-pip
``` 
to upgrade it 
```
python -m pip install -- upgrade pip setuptools wheel
```
to upgrade Python
```
pip install python -- upgrade
```
Additional Packages that are required are: [Numpy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [MatplotLib](https://matplotlib.org/), [Pytorch](https://pytorch.org/), PIL and json.\
You can donwload them using [pip](https://pypi.org/project/pip/)
```
pip install numpy pandas matplotlib pil
```
or [conda](https://anaconda.org/anaconda/python)
```
conda install numpy pandas matplotlib pil
```
In order to intall Pytorch head over to the Pytorch site select your specs and follow the instructions given.

## Command Line Application
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  * Options:
    * Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    * Choose arcitecture (alexnet, densenet121 or vgg16 available): ```pytnon train.py data_dir --arch "vgg16"```
    * Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20 ```
    * Use GPU for training: ```python train.py data_dir --gpu gpu```
    
* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:``` python predict.py input checkpoint ---top_k 3```
    * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```

## Json file
Since the images are saved in folders. A json file is created to match folder names to category names. This helps print the particular category names during prediction. 

## Data and the json file
The data used specifically for this assignemnt are a flower database are not provided in the repository as it's larger than what github allows. Nevertheless, feel free to create your own databases and train the model on them to use with your own projects. The structure of your data should be the following:\
The data need to comprised of 3 folders, test, train and validate. Generally the proportions should be 70% training 10% validate and 20% test.\
Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file. For example if we have the image a.jpj and it is a rose it could be in a path like this /test/5/a.jpg and json file would be like this {...5:"rose",...}.
    

## GPU
Since the model is training on image files a simple cpu is going to take forever to complete the job. Alternatively a GPU can be used in either of the two ways:

1. **Cuda** -- On a local laptop with a NVDIA graphic card CUDA can be used to train the model, this will expedite training time.
2. **Cloud Services** -- One can spin up a GPU on any of the popular cloud services like AWS, GCP, Azure etc. 

For prediction a CPU can be used. 


## Pre-Trained Network
The checkpoint.pth file contains the information of a network trained to recognise 102 different species of flowers. It has been trained with specific hyperparameters thus if you don't set them right the network will fail. In order to have a prediction for an image located in the path /path/to/image using my pretrained model you can simply type ```python predict.py /path/to/image checkpoint.pth```

This project is licensed under the MIT License - see the [LICENSE.md]


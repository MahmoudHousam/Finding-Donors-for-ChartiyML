
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
import json
import PIL
from PIL import Image
import argparse

import functions

parser = argparse.ArgumentParser(description='Predict.py')

parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

exe= parser.parse_args()
path_image= exe.input
number_of_outputs= exe.top_k
device= exe.gpu
path= exe.checkpoint
exe= parser.parse_args()

def main():
    model= functions.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
    probabilities = functions.predict(image_path, model, number_of_outputs, device)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    i=0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Predicting Done!")

    
if __name__== "__main__":
    main()
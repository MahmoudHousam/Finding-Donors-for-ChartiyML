import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import functions

ap = argparse.ArgumentParser(description='Train.py')


parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.3)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=12)
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)



exe = parser.parse_args()
location = exe.data_dir
path = exe.save_dir
lr = exe.learning_rate
structure = exe.arch
dropout = exe.dropout
hidden_layer1 = exe.hidden_units
device = exe.gpu
epochs = exe.epochs

def main():
    
    train_loader, valid_loader, test_loader = functions.dataloaders(location)
    model, optimizer, criterion = functions.nn_class(structure, hidder_layer1, hidden_layer2, output_layer, dropout, lr, power)
    functions.model_processing(model, optimizer, criterion, epochs, print_every, train_loader, power)
    functions.save_checkpoint(model, train_data, path)
    print("Training Done!")


if __name__== "__main__":
    main()
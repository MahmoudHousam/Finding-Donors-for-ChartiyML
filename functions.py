import numpy as np
import json
import PIL
from PIL import Image
from workspace_utils import active_session
import torch
from torch import nn
from torch import tensor
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models



model_arch= {'vgg16':25088, 'densnet121':1024}
#load_flower_names
with open('cat_to_name.json', 'r') as f:
    flower_names= json.load(f)
    

#data_transformation
def dataloaders(location):
    data_dir = location
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {'train_transforms':transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
                    'valid_test_transforms':transforms.Compose([
                        transforms.Resize(225),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                    ])}

    train_data= datasets.ImageFolder(train_dir, transform= data_transforms['train_transforms']),
    valid_data= datasets.ImageFolder(valid_dir, transform= data_transforms['valid_test_transforms']),
    test_data= datasets.ImageFolder(test_dir, transform= data_transforms['valid_test_transforms'])

    train_loader= torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle= True),
    valid_loader= torch.utils.data.DataLoader(valid_data, batch_size= 64, shuffle= True),
    test_loader= torch.utils.data.DataLoader(test_data, batch_size= 64, shuffle= True)
    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader


#building a new classifier for the network
def nn_class(structure= 'vgg16', hidder_layer1= 512, hidden_layer2= 256, output_layer= 102, dropout= 0.3, lr= 0.001, power= 'gpu'):
    '''
    This builds the CNN network classifier that takes the following:
    - CNN Network: (calsss) vgg16 or densnet121
    - Two hidden layers: (int) 512, 256
    - output_layer: (int) 120
    - dropout: (float) 0.3
    - learning_rate: (float) 0.001
    - power: (image processor/str) gpu
    '''
    #structure condition
    if structure == 'vgg16':
        model= models.vgg16(pretrained= True)
        model.name= 'vgg16'
    elif structure == 'densnet121':
        model= models.densenet121(pretrained= True)
        model.name= 'densnet121'
    else:
        print("Only {} or {} are available models".format(list(model_arch.keys())))

    #freezing model's parameters
    for parameter in model.parameters:
        parameter.requires_grad= False
    
    #building classifier
    classifier= nn.sequential(
        nn.Linear(model_arch[structure], hidder_layer1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidder_layer1, hidden_layer2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidder_layer2, output_layer),
        nn.LogSoftmax(dim=1)
    )
    model.classifier= classifier
    criterion= nn.NLLLoss()
    optimizer= optim.Adam(model.classifier.parameters(), lr= lr)
    if torch.cuda.is_available() and power == 'gpu':
        model.cuda()
    else:
        model.cpu()
    return model, model.name, criterion, optimizer


#training model
def model_processing(model, train_loader, valid_loader, criterion, optimizer, epochs= 12, print_every= 5, power= 'gpu'):
    steps= 0
    training_loss= 0
    print("-------------------Your data is proccessed-------------------")
    with active_session():
        for epoch in range(epochs):
            #train the model
            for images, labels in train_loader:
                steps += 1 
                if torch.cuda.is_available() and power == 'gpu':
                    images, labels= images.to('cuda'), labels.to('cuda') 
                optimizer.zero_grad() 
                training_output= model.forward(images)
                tr_loss= criterion(train_output, labels)
                tr_loss.backward() 
                optimizer.step()
                training_loss += tr_loss.item() 


                if steps % print_every == 0:
                    validation_loss= 0
                    accuracy= 0
                    model.eval()
                    with torch.no_grad():
                            #valid the model
                            for images, labels in valid_loader:
                                if torch.cuda.is_available() and power == 'gpu':
                                    images, labels= images.to('cuda'), labels.to('cuda')
                                valid_output= model.forward(images)
                                valid_loss= criterion(valid_output, labels)
                                validation_loss += valid_loss.item()
                                #measure accuracy
                                ps= torch.exp(valid_output)
                                top_p, top_class= ps.topk(1, dim=1)
                                equal= top_class == labels.view(*top_class.shape)
                                #is it safe and accepted to use .data and .max()
                                # equal= (labels.data == ps.max(1)[1]) 
                                accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
                model.train()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {training_loss/steps:.3f}.. "
                      f"Validation loss: {validation_loss/len(valid_loader):.3f}.."
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                
                print("Training Loss Mean: {}".format(np.mean(training_loss/steps)),
                      "Validation Loss Mean: {}".format(np.mean(validation_loss/len(valid_loader))),
                      "Model Accuracy: {}".format(np.mean(accuracy/len(valid_loader))),
                      "-------------------Model has finished-------------------",
                      "The model needs {} epochs and {} steps to train and valid your data".format(epochs, steps), sep= "\n")


#Testing the model
def testing_model(model, test_loader, criterion, power= 'gpu'):
    test_loss = 0
    accuracy= 0
    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available() and power == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output) 
            top_p, top_class= ps.topk(1, dim=1)
            equal= top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
            
            print(f"Validation accuracy: {accuracy/len(test_loader):.3f}")


#saving the model trained
def save_checkpoint(model, train_data, path='checkpoint.pth'):
    checkpoint={
    'architecture': model.name,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'classifier': model.classifier,
    'class_label': train_data.class_to_idx,
    'epochs': epochs,
    'learning_rate': 0.001}
    return torch.save(checkpoint, path)


#loading the model
def load_checkpoint(file):
    checkpoint= torch.load(file)
    model= getattr(torchvision.models, checkpoint['architecture'])(pretrained=True) #returns the value of the named attribute of an object
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_label']
    optimizer.load_state_dict(checkpoint['optimizer'])
    learning_rate = checkpoint['learning_rate']
    return model
model= load_checkpoint('checkpoint.pth')


#image processing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pic= Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    pic_trnasform= transform(pic)
    pic_to_array= np.array(pic_trnasform)
    return pic_trnasform


#predictions 
def predict(image_path, model, topk=5, power= 'gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    pic=process_image(image_path)
    pic= torch.from_numpy(pic).type(torch.FloatTensor)
    pic= pic.unsqueeze_(0)
    load_model= load_checkpoint(model)
    if torch.cuda.is_available() and device =='gpu':
        load_model.to('cuda')
        pic= pic.to('cuda')
        
    load_model.eval()
    
    with torch.no_grad():
        output= load_model.forward(pic_dim)
        ps= torch.exp(output)
        top_probs_list= ps.topk(topk)[0].numpy()[0]
        top_class_list= ps.topk(topk)[1].numpy()[0]
        class_to_idx= load_model.class_to_idx
        idx_to_class= {x: y for y, x in class_to_idx.items()}
        list_of_top_classes= []
        for classes in top_class_list:
            list_of_top_classes += [idx_to_class[classes]]
    return top_probs_list, list_of_top_classes
import torch,torchvision
from torch import nn,optim
from torchvision import transforms,datasets,models
from torchvision.models import densenet121, DenseNet121_Weights,vgg16,VGG16_Weights
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import argparse
import gc
import os
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:64'


parser = argparse.ArgumentParser()
parser.add_argument('--arch', help='models to use [vgg,densenet]',default='densenet121')
parser.add_argument('--learning_rate', help='learning rate',type=float, default=0.001)
parser.add_argument('--hidden_units', help='number of hidden units', type=int, default=500)
parser.add_argument('--epochs', help='epochs',type=int, default=1)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

lr=float(args.learning_rate)
epochs=int(args.epochs)
hidden_units=int(args.hidden_units)
arch=args.arch
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu==True else "cpu")
#device=torch.device("cuda")
#print(lr,epochs,device,arch)

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms ={
    'train':transforms.Compose([transforms.Resize(224),
                               transforms.RandomCrop(224),
                               transforms.RandomRotation(30),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]),
    'valid':transforms.Compose([transforms.Resize((224,224)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'test':transforms.Compose([transforms.Resize((224,224)),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  
}

image_datasets =  {
    'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])}


dataloaders = {
    'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=False),
    'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True)
}
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


if arch=='densenet121':
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    #num_features = model.classifier[-1].in_features    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1',nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(hidden_units, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
else:
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False  
    feature_num = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, hidden_units)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier

#print(model)
print("Training Started")
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)
model.to(device)

gc.collect()
torch.cuda.empty_cache()

steps  = 0
running_loss = 0
print_every  = 50

for epoch in range(epochs):
    
    for inputs, labels in dataloaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss  = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                  f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
            running_loss = 0

print("Training Done\n")
criterion = nn.NLLLoss()
test_loss = 0
accuracy = 0
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
print(f"test loss: {test_loss/len(dataloaders['test']):.3f}.. "
      f"test accuracy: {accuracy/len(dataloaders['test']):.3f}") 

model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'input_size': 1024,
              'output_size': 102,
              'arch': 'densenet121',
              'classifier' : model.classifier,
              'features': model.features,
              'learning_rate': lr,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
             }

torch.save(checkpoint, 'checkpoint.pth')
print("Checkpoint saved as checkpoint.pth")
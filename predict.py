import torch,torchvision
from torch import nn,optim
from torchvision import transforms,datasets,models
from torchvision.models import densenet121, DenseNet121_Weights,vgg16,VGG16_Weights
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
import argparse 
import json
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--image_path',help='Give image path',default='flowers/test/1/image_06764.jpg')
parser.add_argument('--checkpoint',help='Checkpoint path',default='checkpoint.pth')
parser.add_argument('--top_k',help='Classes to display',default='5')
parser.add_argument('--category_names',help='category name file',default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

checkpoint_path=args.checkpoint
image_path=args.image_path
top_k=int(args.top_k)
category_name=args.category_names
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu==True else "cpu")
with open(category_name, 'r') as f:
    cat_to_name = json.load(f)
def Checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(torchvision.models, checkpoint['arch'])(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

model=Checkpoint(checkpoint_path)  
#print(model)


def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image, 'r')
    size = (256, 256)
    pil_image.thumbnail(size)
    width, height = pil_image.size   
    #print(width,height)
    left = (width - 244)/2
    top = (height - 244)/2
    right = (width + 244)/2
    bottom = (height + 244)/2
    pil_image = pil_image.crop((left, top, right, bottom))
    np_image = np.array(pil_image)
    np_image = np_image.astype('float64')
    np_image = np_image / [255,255,255]
    np_image = (np_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    np_image=torch.Tensor(np_image)
    
    return np_image


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to('cpu')
    model.eval()
    with torch.no_grad():
        image = process_image(image_path)
        image.unsqueeze_(0)
        image = image.float()
        outputs = model.forward(image)
        probs, classes = torch.exp(outputs).topk(topk)
        return probs[0].tolist(), classes[0].add(1).tolist()
    # TODO: Implement the code to predict the class from an image file
model=Checkpoint(checkpoint_path) 
prediction=predict(image_path,model,top_k)
probs, classes = predict(image_path,model,top_k)
plant_classes = [cat_to_name[str(cls)] for cls in classes]
#probs = str(round(probs,4) * 100.) + '%'
print("Top {} predictions are:".format(top_k))
for i in range(top_k+1):
    probs[i] = str(round(probs[i],4) * 100) + '%'
    print("{} {}={}".format(i,plant_classes[i],probs[i]))

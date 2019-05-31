# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import torch as t
import torchvision as tv
import os
from  torch.autograd import Variable
import torchvision.transforms as transforms
from vgg_cifar import VGG


if  os.path.isfile('./vggmodel.pt'):
    model = VGG('VGG16')
    print("yes")
    device = t.device("cuda")
    checkpoint = t.load('./vggmodel.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(epoch)
    loss = checkpoint['loss']
    print(loss)
    model.to(device)
#model.load_state_dict(t.load('./model'))
#odel.eval()
use_cuda = t.cuda.is_available()
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
testdata = tv.datasets.CIFAR10(root='./data',train=False,transform=trans,download=False)
testloader = t.utils.data.DataLoader(testdata,batch_size=100,shuffle=True)
correct =0
acc =0
for i,(inputs,labels) in enumerate(testloader):
        inputs, labels = Variable(inputs), Variable(labels)
        #print(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        output = model(inputs)
        y= output.data.max(1)[1]
        #_,y=t.max(output,1)
        correct += y.eq(labels).sum()
print(len(testloader.dataset))
print(correct)
acc =  100. *correct/len(testloader.dataset)
print('Test set:  epoch is {} Accuracy: {}/{} ={:.2f}%'.format(epoch,correct, len(testloader.dataset), acc))
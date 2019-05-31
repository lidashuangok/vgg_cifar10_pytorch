# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import os
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
use_cuda = t.cuda.is_available()
from vgg_cifar import VGG
vggnet = VGG('VGG16')
if use_cuda:
    print(t.cuda.get_device_name())
    vggnet = vggnet.cuda()

#trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
trans = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cri = nn.CrossEntropyLoss()
optimizer = optim.SGD(vggnet.parameters(),lr=0.001,momentum=0.9,weight_decay=5e-4)
traindata = tv.datasets.CIFAR10(root='./data',train=True,transform=trans,download=False)
trainloader = t.utils.data.DataLoader(traindata,batch_size=100,shuffle=True)
def train():
    if os.path.isfile('./vggmodel.pt'):
        print("yess")
        device = t.device("cuda")
        print(t.cuda.get_device_name())
        checkpoint = t.load('./vggmodel.pt')
        vggnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        vggnet.to(device)
    else:
        epoch = 0
    for epoch in range(epoch,epoch+20):
        for i,(inputs,labels) in enumerate(trainloader):
            #print(labels)
            inputs, labels = Variable(inputs), Variable(labels)
            #print(labels)
            inputs=inputs.cuda()
            labels =labels.cuda()
            optimizer.zero_grad()
            output = vggnet(inputs)
            #print(output)
            loss = cri(output,labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 or (i + 1) == len(trainloader):
                print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}'.format(epoch, i+ 1, loss ))
    print(epoch)
    print(loss)
    #t.save(net,'./model.pt')
    t.save({
        'epoch': epoch,
        'model_state_dict': vggnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, './vggmodel.pt')
if __name__ == '__main__':
    train()

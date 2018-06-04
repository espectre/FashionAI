import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
import argparse
import numpy as np 
from torch.optim.lr_scheduler import *

from model.resnet import resnet101
from data_pre.FashionAI import fashion 

parser=argparse.ArgumentParser()
parser.add_argument('--workers',type=int,default=2)
parser.add_argument('--batchSize',type=int,default=64)
parser.add_argument('--nepoch',type=int,default=11)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--gpu',type=str,default='7')
parser.add_argument('--attr',type=str,default='collar_design_labels')
opt=parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

transform_train=transforms.Compose([
	transforms.Resize((256,256)),
	transforms.RandomCrop((224,224)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

transform_val=transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

trainset=fashion('/home/yhf/Challenge/FashionAI/STL_FashionAI/data/2base/Annotations/sum_labels.csv',transform_train,opt.attr,train=True)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.workers)
valset=fashion('/home/yhf/Challenge/FashionAI/STL_FashionAI/data/2base/Annotations/sum_labels.csv',transform_val,opt.attr,train=False)
valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.workers)
AttrNum={
	"coat_length_labels":8,
	"collar_design_labels":5,
	"lapel_design_labels":5,
	"neck_design_labels":5,
	"neckline_design_labels":10,
	"pant_length_labels":6,
	"skirt_length_labels":6,
	"sleeve_length_labels":9
}

model=resnet101(pretrained=True)
model.fc=nn.Linear(2048,AttrNum[opt.attr])
model.cuda()
optimizer=torch.optim.SGD(model.parameters(),lr=opt.lr,momentum=0.9,weight_decay=5e-4)
scheduler=StepLR(optimizer,step_size=3)
criterion=nn.CrossEntropyLoss()
criterion.cuda()

def train(epoch):
	print('\nTrain Epoch:%d' % epoch)
	scheduler.step()
	model.train()
	for batch_idx, (img,label) in enumerate(trainloader):
		image=Variable(img.cuda())
		label=Variable(label.cuda())
		optimizer.zero_grad()
		out=model(image)
		loss=criterion(out,label)
		loss.backward()
		optimizer.step()
		if batch_idx%20==0:
			print("Epoch: %d [%d:%d] loss: %f" % (epoch,batch_idx,len(trainloader),loss.mean()))

def val(epoch):
	print('\nTest Epoch:%d'%epoch)
	model.eval()
	total=0
	correct=0
	for batch_idx, (img,label) in enumerate(valloader):
		image=Variable(img.cuda(),volatile=True)
		label=Variable(label.cuda())
		out=model(image)
		_,predict=torch.max(out.data,1)
		total+=image.size(0)
		correct+=predict.eq(label.data).cpu().sum()
	print("Acc:%f" % ((1.0*correct)/total))

for epoch in range(opt.nepoch):
	train(epoch)
	val(epoch)
torch.save(model.state_dict(),'ckp/model_task_%s.pth' % opt.attr)

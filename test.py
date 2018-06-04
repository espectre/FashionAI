import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
import argparse
import numpy as np 
from torch.optim.lr_scheduler import *
import csv

from model.resnet import resnet101
from data_pre.fashionTEST import fashiontest 

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=128)
parser.add_argument('--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--gpu', type=str, default='7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--attr', type=str,default='collar_design_labels')
opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

transform_test = transforms.Compose([
    transforms.Scale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

AttrNum = {
    'coat_length_labels':8,
    'collar_design_labels':5,
    'lapel_design_labels':5,
    'neck_design_labels':5,
    'neckline_design_labels':10,
    'pant_length_labels':6,
    'skirt_length_labels':6,
    'sleeve_length_labels':9,
}

testset=fashiontest('/home/yhf/Challenge/FashionAI/ex_STL_FashionAI/data/2rank/Tests/question.csv',transform_test,opt.attr)
testloader=torch.utils.data.DataLoader(testset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.workers)
model=resnet101(pretrained=True)
model.fc=nn.Linear(2048,AttrNum[opt.attr])
model.load_state_dict(torch.load('ckp/model_task_%s.pth'%opt.attr))
model.cuda()
model.eval()
results=[]

for image,addrs in testloader:
	image=Variable(image.cuda(),volatile=True)
	out=model(image)
	out=np.exp(out.cpu().data.numpy()).tolist()
	results.extend([[j,opt.attr,";".join([str(ii) for ii in i])] for (i,j) in zip(out,addrs)])

eval_csv=os.path.join(os.path.expanduser('.'),'deploy',opt.attr+'_eval.csv')
with open(eval_csv,'w',newline='') as f:
	writer=csv.writer(f,delimiter=',')
	for x in results:
		writer.writerow(x)

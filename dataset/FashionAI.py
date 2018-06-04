from PIL import Image
import torch.utils.data as data
import numpy as np 
import os
import csv
import random

def pil_loader(path):
	with open(path,'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def make_img(label_file,label,train=True):
	img=[]
	with open(label_file) as f:
		count=0
		reader=csv.reader(f)
		for row in reader:
			if train==True and row[1]==label and count%10<=7:
				img.append((os.path.join(os.path.expanduser('.'),'data/2base',row[0]),int(row[2].find('y'))))
			elif train==False and row[1]==label and count%10>7:
				img.append((os.path.join(os.path.expanduser('.'),'data/2base',row[0]),int(row[2].find('y'))))
			count+=1
			#if count>=640:
			#	break
	return img

class fashion(data.Dataset):
	def __init__(self,label_file,transform,label,train=True):
		self.train=train
		self.img=make_img(label_file,label,self.train)
		self.length=len(self.img)
		self.transform=transform

	def __len__(self):
		return self.length

	def __getitem__(self,index):
		image=pil_loader(self.img[index][0])
		if self.transform is not None:
			image=self.transform(image)
		return image,self.img[index][1]

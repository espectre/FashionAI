from PIL import Image
import os
import torch.utils.data as data
import numpy as np 
import random
import csv

def pil_loader(path):
	with open(path,'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def make_img(label_file,label):
	img=[]
	with open(label_file) as f:
		reader=csv.reader(f)
		for row in reader:
			if row[1]==label:
				img.append(os.path.join('/home/yhf/Challenge/FashionAI/ex_STL_FashionAI/data/2rank',row[0]))
	return img

class fashiontest(data.Dataset):
	def __init__(self,label_file,transform,label):
		self.img=make_img(label_file,label)
		self.length=len(self.img)
		self.transform=transform

	def __len__(self):
		return self.length

	def __getitem__(self,index):
		image=pil_loader(self.img[index])
		if self.transform is not None:
			image=self.transform(image)
		return image,self.img[index]

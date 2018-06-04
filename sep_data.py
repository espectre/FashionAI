from PIL import Image
import os
import torch.utils.data as data
import numpy as np 
import random
import csv

label_file="data/base/Annotations/sum_labels.csv"
img_train=[]
img_val=[]
with open(label_file) as f:
	count=0
	reader=csv.reader(f)
	for row in reader:
		if count%10<=7:
			img_train.append((os.path.join("home/yhf/Challenge/FashionAI/ex_STL_FAshionAI/data/base",row[0]),row[1],row[2]))
		else:
			img_val.append((os.path.join("home/yhf/Challenge/FashionAI/ex_STL_FAshionAI/data/base",row[0]),row[1],row[2]))
		count+=1
train_csv=os.path.join(os.path.expanduser('.'),'data/base/Annotations','label_train.csv')
with open(train_csv,'w') as f:
	writer=csv.writer(f,delimiter=',')
	for x in img_train:
		writer.writerow(x)


val_csv=os.path.join(os.path.expanduser('.'),'data/base/Annotations','label_val.csv')
with open(val_csv,'w') as f:
	writer=csv.writer(f,delimiter=',')
	for x in img_val:
		writer.writerrow(x)
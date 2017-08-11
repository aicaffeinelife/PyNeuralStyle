import sys, os 
import numpy as np 
import argparse 

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.autograd import Variable 
import torchvision.transforms as tf 
import torchvision.models as models
from utils import check_dirs, imgloader, imgshow 
from loss import ContentLoss, GramMatrix, StyleLoss


parser = argparse.ArgumentParser()

# basic options 
parser.add_argument('--photo_dir', type=str, required=True, help='Path to the actual photograph')
parser.add_argument('--art_dir',   type=str, required=True, help='Path to the artwork')
parser.add_argument('--img_sz',    type=int, default=200,   help='The size of the image')
parser.add_argument('--out_dir',   type=str, required=True, help='Save path for the image')
parser.add_argument('--cuda',      type=bool,default=True,  help='Use GPU if available' )

# neural style options
parser.add_argument('--w_content', type=float, default=1, help='Content weight')
parser.add_argument('--w_style',   type=float, default=1e3, help='Style weight')



args = parser.parse_args()

check_dirs(args.photo_dir)
check_dirs(args.art_dir)
check_dirs(args.out_dir, make=True)

ip_tfs = tf.Compose([
			tf.ToTensor()
		])

pil_tf = tf.ToPILImage()


vgg19 = models.vgg19(pretrained=True).features 
model = nn.Sequential()
gm = GramMatrix()
dtype = torch.FloatTensor 
if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor

if args.cuda:
	vgg19 = vgg19.cuda()
	model = model.cuda()
	gm = gm.cuda()


content_layers = ['conv_4'] # The fourth convolutional layer's activation to serve as the content
style_layers = ['conv_1', 'conv_2','conv_3', 'conv_4','conv_5'] # The list of layers contributing to the style representations

cl_list = [] # to compute the common loss term
sl_list = []

photo = imgloader(args.photo_dir, ip_tfs, 512).type(dtype)
art = imgloader(args.art_dir, ip_tfs, 512).type(dtype)
print(photo.size())
print(art.size())
assert(photo.size() == art.size())
i = 1
# build the neural network to optimize
for layer in list(vgg19):
	if isinstance(layer, nn.Conv2d):
		conv = 'conv_'+str(i)
		model.add_module(conv, layer)

		if conv in content_layers:
			target = model.forward(photo).clone()
			cl = ContentLoss(args.w_content, target)
			model.add_module('content_loss_'+str(i), cl)
			cl_list.append(cl)

		if conv in style_layers:
			fm = model.forward(art).clone()
			target_fm = gm.forward(fm)
			sl = StyleLoss(args.w_style, target_fm)
			model.add_module('style_loss_'+str(i), sl)
			sl_list.append(sl)
		i += 1

	if isinstance(layer, nn.ReLU):
		relu = 'relu_'+str(i)
		model.add_module(relu, layer)

	if isinstance(layer, nn.MaxPool2d):
		mpool = 'maxpool_'+str(i)
		model.add_module(mpool, layer)
	

print(model)

# setup inputs and optimizers 
in_data = imgloader(args.photo_dir, ip_tfs, 512).type(dtype)

in_data = nn.Parameter(in_data.data)
optimz = optim.LBFGS([in_data])

num_iters = 300 
iters = [0] 

while iters[0] <= num_iters:
	
	def optim_step():
		in_data.data.clamp_(0, 1)
		optimz.zero_grad()
		model.forward(in_data)
		sl = 0
		cl = 0

		for c in cl_list:
			cl += c.backward()
		for s in sl_list:
			sl += s.backward()
		iters[0] += 1	
		if iters[0] % 5 == 0:
			print("Content loss: %2.3f"%(cl.data[0]))
			print("Style loss:%2.3f"%(sl.data[0]))
		return cl + sl 

	optimz.step(optim_step)

in_data.data.clamp_(0, 1)
imgshow(in_data, pil_tf, 512,args.out_dir)






















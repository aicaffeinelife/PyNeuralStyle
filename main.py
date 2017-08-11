import sys, os 
import argparse
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torchvision.models as models 
import torchvision.transforms as tfs

parser = argparse.ArgumentParser()
# basic options
parser.add_argument('--ip_img', type=str, required=True, help='the input photo')
parser.add_argument('--art_img', type=str, required=True, help='the artwork')
parser.add_argument('--img_size', type=int, default=512, help='resize image to same height & width')

# neural style options 
parser.add_argument('--content_weight', type=float, default=5e0, help='alpha for L_content')
parser.add_argument('--style_weight', type=float, default=1e2,  help='beta for L_style')

# training params
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
parser.add_argument('--print_every', type=int, default=100, help='print every iters')
parser.add_argument('--save_every', type=int, default=100, help='save output img')
parser.add_argument('--num_iters', type=int, default=10000, help='num iterations to train')

#output 
parser.add_argument('--out_path', type=str, required=True, help='Path to save directory')

# misc 
parser.add_argument('--debug', type=bool, default=False, help='print debug log')

args = parser.parse_args() 

# helper function to check dirs 
def check_dirs(ip_dir, make=False):
	if not os.path.exists(ip_dir) and not make:
		print("%s doesn't exist, check path"%(ip_dir))
		sys.exit(1)
	elif not os.path.exists(ip_dir) and make:
		os.makedirs(ip_dir)




# Net creation functions 

def remove_max_pool(net): 
	lst = []
	for m in net.children():
		if isinstance(m, nn.MaxPool2d):
			lst.append(nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)))
		else:
			lst.append(m)
	return nn.Sequential(*lst)


	
def make_net_from_VGG(p_net):
	net = None 
	if p_net == 'VGG19':
		orig_net = models.vgg19(pretrained=True)
	elif p_net == 'VGG16':
		orig_net = models.vgg16(pretrained=True)
	else:
		print("Accepted pre-trained nets: VGG{16,19}")

	if args.debug:
		print("----- original net ----- ")
		print(orig_net)
	net = remove_max_pool(orig_net.features)
	print("----- new net ----- ")
	print(net)
	return net









if __name__ == '__main__':
	make_net_from_VGG('VGG19')












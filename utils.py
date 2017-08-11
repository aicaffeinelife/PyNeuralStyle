import sys, os
import torch 
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt 
from torch.autograd import Variable

def check_dirs(Dir, make=False):
	if not os.path.exists(Dir) and not make:
		raise Exception('%s doesnt exist'%(Dir))
	elif not os.path.exists(Dir) and make:
		os.makedirs(Dir)




def imgloader(img_path, transform, img_sz):
	img = Image.open(img_path).resize((img_sz, img_sz))
	timg = transform(img)
	return Variable(timg).unsqueeze(0)

def imgshow(img_tensor, pil_transform, img_sz, o_dir, save=True):
	img = img_tensor.data.cpu().view(3, img_sz, img_sz)
	img = pil_transform(img)
	if save:
		img.save(os.path.join(o_dir + 'gen_img.jpg'))
	else:
		plt.imshow(img)


	



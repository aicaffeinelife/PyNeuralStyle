import torch 
import torch.nn as nn 
from torch.autograd import Variable


class ContentLoss(nn.Module):
	"""
	ContentLoss: Computes the "content loss" between 
	the feature representations of a white noise image at
	layer_i vs the feature representations of the actual image
	at layer_i. 

	Here weight is the content_loss weight (or alpha in the paper)

	"""
	def __init__(self, weight, target):
		super(ContentLoss, self).__init__()
		self.target = target.detach() * weight
		self.weight = weight 
		self.criteria = nn.MSELoss()

	def forward(self, inp):
		self.loss = self.criteria.forward(inp * self.weight, self.target)
		self.output = inp
		return self.output

	def backward(self):
		self.loss.backward(retain_variables=True)
		return self.loss

class GramMatrix(nn.Module):
	"""
	GramMatrix: Calculate the correlation between 
	the vectorized feature maps in a conv layer activation 

	The output of the feature map is reshaped to KxN where
	K: bsxnum_ch and N: hxw 

	The gram matrix is calculated as the inner product of the fm
	"""
	def forward(self, inp):
		bs, ch, h, w = inp.size()
		k = bs*ch 
		n = h*w 
		inp = inp.view(k,n)
		gm = torch.mm(inp, inp.t())
		return gm.div(k*n)


class StyleLoss(nn.Module):
	"""
	StyleLoss: Computes the "style loss" between 
	the white noise and the target artwork. 
	The style loss is calculated as the MSELoss 
	between the gram matrices of the random image 
	and the artwork

	Here the weight is style_loss weight(or beta in the paper)
	"""
	def __init__(self, weight, target):
		super(StyleLoss, self).__init__()
		self.target = target.detach() * weight 
		self.weight = weight 
		self.gram = GramMatrix()
		self.criteria = nn.MSELoss()

	def forward(self, inp):
		self.output = inp.clone() 
		self.gm_inp = self.gram.forward(inp)
		self.gm_inp.mul_(self.weight)
		self.loss = self.criteria.forward(self.gm_inp, self.target)
		return self.output

	def backward(self):
		self.loss.backward(retain_variables=True)
		return self.loss

		




		



	





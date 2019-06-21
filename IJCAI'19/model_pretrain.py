from block import *

class Encoder(nn.Module):
	def __init__(self, name):
		super(Encoder, self).__init__()
		if 'UnetED' in name:
			self.encode = Encode(name)
		if 'DuoED' in name:
			self.inst_encode = Encode(name)

	def forward(self, _input, Xavg, Xstd, name):
		def get_inst_x(x,avg,std):
			xs = x.size()
			avg = avg.view(1, avg.size()[0],1,1).repeat(xs[0], 1, xs[2], 1)
			std = std.view(1, std.size()[0],1,1).repeat(xs[0], 1, xs[2], 1)
			x = (x - avg)/std
			return x
		
		x = _input.unsqueeze(3) 
		x = get_inst_x(x,Xavg,Xstd)
		x = x.permute(0,3,1,2)

		l1,l2,l3,l4,l5 = torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1)
		if 'UnetED' in name:
			vec,s,c = self.encode(x,name)
		
		if 'DuoED' in name:
			vec, s = self.inst_encode(x,name)
	
		return vec

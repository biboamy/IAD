import torch, os
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '3' # change
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score 
from skimage.measure import block_reduce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats

nums_label = 10

def parse_sidd_data():
	result=[]
	f = np.load('../disentangle/track_wise_outputs_raw.npz')
	meddata = os.listdir('../../instrument/medley/feature/inst_pred/test_mix_12_inst_label/')
	mixdata = os.listdir('../../instrument/mix_secret/feature/test_mix_12_inst_label/')
	outputs = f['outputs'].item()
	result_stack = []
	for output in outputs.keys():
		try:
			if output+'.npy' in meddata or output+'.npy' in mixdata:
				tar = outputs[output]['targets']
				pre = outputs[output]['predictions']
				tar = np.delete(tar,[0,1,3,5,10,11,12,14,15,16,17],1)
				pre = np.delete(pre,[0,1,3,5,10,11,12,14,15,16,17],1)
				result_stack.append(roc_auc_score(tar[:,:].flatten(),pre[:,:].flatten(),average='micro'))
			result_stack.append(roc_auc_score(t.flatten(),p.flatten(),average='micro'))
		except Exception as e: pass#print(e)

	return result_stack

class conv_block(nn.Module):
    def __init__(self, inp, out, kernal, pad):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(inp, out, kernal, padding=pad)
        self.batch = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.batch(self.conv(x)))
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.head = nn.Sequential(
        	conv_block(128,128*2,(5,3),(0,1)),
        	conv_block(128*2,128*3,(1,3),(0,1)),
        	conv_block(128*3,128*3,(1,3),(0,1)),
        	conv_block(128*3,128,(1,3),(0,1))
        )

        self.head2 = nn.Sequential(
        	nn.Linear(128,1024),
        	nn.Linear(1024,nums_label)
        )
 
    def forward(self, _input):
    	oup = self.head(_input).squeeze(2)
    	
    	oup = self.head2(oup.permute(0,2,1))
    	oup = F.max_pool2d(oup.permute(0,2,1),(1,2),(1,2))

    	return oup

# UnetAE_preIP_preRoll 
# UnetAE_preIP_prePP_prePNZ_preRoll 
# UnetAE_preIP_prePP_prePNN_preRoll 
# DuoAE_preIP_prePP_preRoll
# DuoAE_preIP_preINZ_prePP_prePNZ_preRoll
# DuoAE_preIP_preINN_prePP_prePNN_preRoll
# twoStep
# MTAN
name = 'DuoAE_preIP_preINZ_prePP_prePNZ_preRoll'

for i in range(0,10):
	load_name = name + '_test'
	model = Net().cuda()
	model_dict = model.state_dict()
	save_dic = torch.load('./model/2019417/'+name+'/e_'+str(i+1))
	model_dict.update(save_dic['state_dict']) 
	model.load_state_dict(model_dict)
	model.eval()
	

	def test_song():
		from scipy import stats
		result_stack = []
		for f in os.listdir('feature/latent_inst_'+str(nums_label)+'/'+load_name+'/'):
			tar = np.load('feature/label_test_'+str(nums_label)+'/'+f)
			pre = np.load('feature/latent_inst_'+str(nums_label)+'/'+load_name+'/'+f)
			pre = np.pad(pre,((0,0),(0,0),(0,0),(0,1)),'constant',constant_values=0)
			tar = block_reduce(tar, block_size=(1, 1, 30), func=np.max)[:,:,:-1]
			pre = torch.from_numpy(pre).float().cuda()
			pre = model(pre)
			pre = torch.sigmoid(pre).data.cpu().numpy().squeeze()

			idx = [0,1,2,7,8,9]

			try:
				result = roc_auc_score(tar[:,idx,:].flatten(), pre[:,idx,:].flatten(),average='micro')
				result_stack.append(result)
			except: pass

		result_us = np.array(result_stack)
		result_sid = np.array(parse_sidd_data())
		print(len(result_sid),len(result_us))

		'''
		result_us.sort()
		hmean = np.mean(result_us)
		hstd = np.std(result_us)
		pdf = stats.norm.pdf(result_us, hmean, hstd)
		plt.plot(result_us, pdf)
		plt.savefig('test.png')
		'''
		t2, p2 = stats.ttest_ind(result_us,result_sid,equal_var=False)
		print("t = " + str(t2))
		print("p = " + str(p2))
	test_song()

	def test_instrument():
		label, data = [],[]
		for f in os.listdir('feature/latent_inst_'+str(nums_label)+'/'+load_name+'/'):
			label.append(np.load('feature/label_test_'+str(nums_label)+'/'+f))
			xi = np.load('feature/latent_inst_'+str(nums_label)+'/'+load_name+'/'+f)
			#xp = np.load('feature/latent_pitch/'+load_name+'/'+f)
			data.append(xi)	
		label = np.concatenate(label,0)
		data = np.concatenate(data,0)

		#data = np.pad(data.reshape(-1,640,19),((0,0),(0,0),(0,1)),'constant',constant_values=0)
		data = np.pad(data,((0,0),(0,0),(0,0),(0,1)),'constant',constant_values=0)
		label = block_reduce(label, block_size=(1, 1, 30), func=np.max)[:,:,:-1]
		data = torch.from_numpy(data).float().cuda()
		pred = model(data)
		pred = torch.sigmoid(pred).data.cpu().numpy()
		result_stack = []
		for j in [0,1,2,7,8,9]:
			try:
				result = roc_auc_score(label[:,j,:].flatten(), pred[:,j,:].flatten(),average='micro')
				print(round(result,3))
				result_stack.append(result)
			except: pass
		print(i,'sum:%f',round(sum(result_stack)/6,3))
	test_instrument()
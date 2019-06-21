import torch, os, sys, librosa
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
from torch.utils.data import Dataset
from torch.autograd import Variable
from model_pretrain import *

nums_label = 10

def load_te_mp3(name):
    def logCQT(file):
        sr = 16000
        y, sr = librosa.load(file,sr=sr)
        cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=27.5, n_bins=88, bins_per_octave=12)
        return ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0

    def chunk_data(f):
        s = int(16000*10/512)
        num = 88
        xdata = np.transpose(f)
        x = [] 
        xdata = xdata[:int(len(xdata)/s)*s,:]
        for i in range(int(len(xdata)/s)):
            data=xdata[i*s:i*s+s]
            x.append(data)
        return np.array(x)

    x = logCQT('mp3/'+name)
    x = chunk_data(x)
    x = np.transpose(x,(0, 2, 1))
    return x

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
        oup = F.max_pool2d(oup.permute(0,2,1),(1,2),(1,2),(0,1))
        return oup

# UnetAE_preIP_prePP_prePNZ_preRoll -> UnetED
# DuoAE_preIP_preINZ_prePP_prePNZ_preRoll -> DuoED

def main(argv):
    audio_name = argv[1]
    model_name = argv[2]

    # load data
    Xte = load_te_mp3(audio_name)
    Xte = torch.from_numpy(Xte).float()
    s = Xte.shape
    Xavg, Xstd = np.load('cqt_avg_std.npy')
    Xavg, Xstd = Variable(torch.from_numpy(Xavg).float()),Variable(torch.from_numpy(Xstd).float())
    print ('finishing loading dataset')

    # load model
    #encoder
    encoder = Encoder(model_name)
    model_dict = encoder.state_dict()
    save_dic = torch.load('./models/encoder/'+model_name,map_location='cpu')
    if 'UnetED' in model_name:
        pretrained_dict = {k: v for k, v in save_dic['state_dict'].items() if (k in model_dict and 'encode' in k)}    
    if 'DuoED' in model_name:
        pretrained_dict = {k: v for k, v in save_dic['state_dict'].items() if (k in model_dict and 'inst_encode' in k)}
    model_dict.update(pretrained_dict) 
    encoder.load_state_dict(model_dict)

    #classifier
    classifier = Net()
    model_dict = classifier.state_dict()
    save_dic = torch.load('./models/classifier/'+model_name,map_location='cpu')
    model_dict.update(save_dic['state_dict']) 
    classifier.load_state_dict(model_dict)
    print ('finishing loading model')

    # start predicting
    print ('start predicting...')
    classifier.eval()   
    encoder.eval() 
    data = Variable(Xte)
    data = encoder(data,Xavg, Xstd, model_name)

    pred = classifier(data)
    pred_inst = torch.sigmoid(pred).data.cpu().numpy()
    pred_inst = np.transpose(pred_inst, (1, 0, 2)).reshape((10,-1))
    pred_inst = np.delete(pred_inst,[3],axis=0)
    np.save('output_data/inst/'+audio_name[:-4]+'.npy',pred_inst)

    # plot image
    plt.figure(figsize=(10,3))
    plt.yticks(np.arange(9), ('Piano', 'Acoustic Guitar', 'Electrical Guitar', 'Trumpet', 'Saxophone',\
                              'Bass', 'Violin', 'Cello', 'Flute'))
    plt.imshow(pred_inst, interpolation='nearest', aspect='auto')
    plt.savefig("output_data/pic/"+audio_name[:-4]+".png")

if __name__ == "__main__":
    main(sys.argv)


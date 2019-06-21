from model import *
#from model_baseline import *
import os
from math import sqrt
import numpy as np # change
import SharedArray as sa
from skimage.measure import block_reduce
from sklearn.metrics import roc_auc_score 
import matplotlib, librosa, h5py, mir_eval
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_freq_grid():
    freq_grid = librosa.cqt_frequencies(88, 27.5, 12)
    return freq_grid

def get_time_grid(n_time_frames):
    time_grid = librosa.core.frames_to_time(range(n_time_frames), sr=16000, hop_length=512)
    return time_grid

def matrix_to_mirinp(tar,thresh):
    fre = get_freq_grid()
    time = get_time_grid(tar.shape[1])
    idx = np.where(tar > thresh)
    est_freqs = [[] for _ in range(len(time))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(fre[f])
    est_freqs = [np.array(lst) for lst in est_freqs]
    return time, est_freqs

def eval_pitch(pre_midi,tar_midi):
    ref_times, ref_freqs = matrix_to_mirinp(tar_midi,0)
    est_times, est_freqs = matrix_to_mirinp(pre_midi,0)
    scores = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs)
    return scores['Accuracy']

def chunk_data(f):
    s = int(16000*10/512)
    num = 88
    xdata = np.transpose(f)
    x = [] 
    xdata = xdata[:int(len(xdata)/s)*s,:]
    for i in range(int(len(xdata)/s)):
        data=xdata[i*s:i*s+s]
        x.append(data.T)
    return np.array(x)

def load_te_mp3(name):
    def logCQT(file):
        sr = 16000
        y, sr = librosa.load(file,sr=sr)
        cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=27.5, n_bins=88, bins_per_octave=12)
        return ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0

    x = logCQT('../mp3/'+name)
    x = chunk_data(x)
    x = np.transpose(x,(0, 2, 1))
    return x

def test_audio(stft):
    print(stft.shape)
    reconst = griffin_lim(stft,2048,512,1000)
    librosa.output.write_wav('../output_data/audio/test.wav', reconst, 16000)

# UnetAE_preIP_preRoll -> UnetED    
# UnetAE_preIP_prePP_prePNZ_preRoll 
# UnetAE_preIP_prePP_prePNN_preRoll 
# DuoAE_preIP_prePP
# DuoAE_preIP_preINZ_prePP_prePNZ
# DuoAE_preIP_prePP_preRoll
# DuoAE_preIP_preINZ_prePP_prePNZ_preRoll
# DuoAE_preIP_preINN_prePP_prePNN_preRoll

name = 'UnetAE_preIP_prePP_prePNZ_preRoll'
for i in range(0,2):
    model = Net(name).cuda()
    model_dict = model.state_dict()
    model.eval()
    save_dic = torch.load('../data/model/2019214/'+name+'/e_'+str(i+1)) 
    #save_dic = torch.load('./'+name+'_e_'+str(i+1)) 
    model_dict.update(save_dic['state_dict']) 
    model.load_state_dict(model_dict)
    Xavg, Xstd = np.load('../../data/cqt_avg_std.npy')
    Xavg, Xstd = torch.from_numpy(Xavg).float().cuda(),torch.from_numpy(Xstd).float().cuda()

    def test_inst():
        pre_stack, pre_stack2, tar_stack = [],[],[]
        choose = 'test'
        save_name = name + '_' + choose
        if not os.path.exists('../../train_inst/feature/latent_inst_10/'+save_name+'/'):
            os.makedirs('../../train_inst/feature/latent_inst_10/'+save_name+'/')
        # melody ex
        '''
        for f in os.listdir('../../../instrument/medley/feature/melody_pred/cqt_'+choose+'/'):
            cqt = np.load('../../../instrument/medley/feature/melody_pred/cqt_'+choose+'/'+f)
            Xte = torch.from_numpy(chunk_data(cqt)).float().cuda()
            pred = model(Xte, Xavg, Xstd, name, False)
            np.save('../../train_pitch/feature/latent_pitch/'+save_name+'/'+f,pred[5].data.cpu().numpy())

        '''
        #mixing secret
        for f in os.listdir('../../../instrument/mix_secret/feature/'+choose+'_mix_10_inst_label/'):
            try:
                label = np.load('../../../instrument/mix_secret/feature/'+choose+'_mix_10_inst_label/'+f)
                cqt = np.load('../../../instrument/mix_secret/feature/'+choose+'_mix_all_inst_CQT/'+f)
           
                Xte = torch.from_numpy(chunk_data(cqt)).float().cuda()
                pred = model(Xte, Xavg, Xstd, name, False)
                #if not np.all(label==0): 
                #    np.save('../../train_inst/feature/latent_inst_10/'+save_name+'/'+f,pred[5].data.cpu().numpy())
             
                pred2 = torch.sigmoid(pred[0]).data.cpu().numpy()
                #pred = torch.sigmoid(pred[4])
                #pred = pred.sum(2)
                #pred = pred/pred.max()
                #pred = pred.data.cpu().numpy()
                
                #pre_stack.append(block_reduce(pred, block_size=(1, 1, 15), func=np.max))
                pre_stack2.append(block_reduce(pred2, block_size=(1, 1, 15), func=np.max))
                tar_stack.append(block_reduce(chunk_data(label), block_size=(1, 1, 30), func=np.max))
                
            except Exception as e:print(e)

        # medleydb
        for f in os.listdir('../../../instrument/medley/feature/inst_pred/'+choose+'_mix_10_inst_label/'):
            try:
                label = np.load('../../../instrument/medley/feature/inst_pred/'+choose+'_mix_10_inst_label/'+f)
                cqt = np.load('../../../instrument/medley/feature/inst_pred/'+choose+'_mix_all_inst_CQT/'+f)
         
                Xte = torch.from_numpy(chunk_data(cqt)).float().cuda()
                pred = model(Xte, Xavg, Xstd, name, False)
                #if not np.all(label==0): 
                #    np.save('../../train_inst/feature/latent_inst_10/'+save_name+'/'+f,pred[5].data.cpu().numpy())
             
                pred2 = torch.sigmoid(pred[0]).data.cpu().numpy()
                #pred = torch.sigmoid(pred[4])
                #pred = pred.sum(2)
                #pred = pred/pred.max()
                #pred = pred.data.cpu().numpy()

                #pre_stack.append(block_reduce(pred, block_size=(1, 1, 15), func=np.max))
                pre_stack2.append(block_reduce(pred2, block_size=(1, 1, 15), func=np.max))
                tar_stack.append(block_reduce(chunk_data(label), block_size=(1, 1, 30), func=np.max))
               
            except:pass
        
        #pre_stack=np.concatenate(pre_stack)
        pre_stack2=np.concatenate(pre_stack2)
        tar_stack=np.concatenate(tar_stack)
     
        result_stack= [] 
        result_stack2= [] 
        for j in [0,1,2,7,8,9]:
            result = roc_auc_score(tar_stack[:,j,:].flatten(), pre_stack[:,j,:].flatten(),average='micro')
            result2 = roc_auc_score(tar_stack[:,j,:].flatten(), pre_stack2[:,j,:].flatten(),average='micro')
            print('result:%f',round(result,3))
            print('result2:%f',round(result2,3))
            result_stack.append(result)
            result_stack2.append(result2)
        print(i,'sum:%f',round(sum(result_stack)/6,3),round(sum(result_stack2)/6,3))


    def test_roll():
        data_name = 'musescore_te'
        tedata = h5py.File('../../ex_data/musescore500_10inst/te2.h5', 'r')
        tedata_s = h5py.File('../../ex_data/musescore500_10inst/te2_stream.h5', 'r')
        Xte = tedata['x'][:]
        Ytr_s = tedata_s['y'][:]
        #Xte = sa.attach('shm://%s_Xte'%(data_name))
        #Ytr_s = sa.attach('shm://%s_Yte_stream'%(data_name))
        Xte=torch.from_numpy(Xte).float().cuda().squeeze()
        pred_list = []
        for y in range((int(len(Xte)/100))+1):
            pred = model(Xte[100*y:100*y+100], Xavg, Xstd, name, False)
            pred = (F.sigmoid(pred[4]).data.cpu().numpy())#*torch.sigmoid(pred[0]).unsqueeze(2).repeat(1,1,pred[4].size()[2],1))
            pred_list.append(pred)
        pred = np.concatenate(pred_list,0)-0.8
        pred[pred>0]=1
        pred[pred<=0]=0
        score = []
        for i in range(10):
            for j in range(len(pred)):
                if not np.all(Ytr_s[j,i]==0):
                    s = eval_pitch(pred[j,i],Ytr_s[j,i])
                    score.append(s)
                    #print(j,s,pred[j,i].max(),Ytr_s[j,i].max())
        print(i,round(sum(score)/len(score),3))

    test_inst()
print(name)

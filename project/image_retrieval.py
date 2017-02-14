# -*- coding: utf-8 -*-
'''
Created on 2017年2月9日
@author: RenaiC
'''

import numpy as np
import h5py
import pickle 
import timeit
from Project4_util import performance_evaluation
start_time = timeit.default_timer()
Top_K = [10, 20, 50, 100] 
Top_K = 100
h5f = h5py.File('featsCNN.h5','r')
feats = h5f['dataset_1'][:]
h5f.close()
imgName = []
imgType = []
# imgNames = h5f['dataset_2'][:]
with open('imgNames.pkl','r') as f:
    imgNames = pickle.load(f)

for img in imgNames: # split into {type,name}
    imgName.append(img.split()[1])
    imgType.append(img.split()[0])

time_elapsed = []

precision={} 
recall={} 
f1={}
MRR={}

for itype in np.unique(imgType):
    if itype != 'clutter':  # exclue clutter one
        precision[itype]=[] 
        recall[itype]=[]
        f1[itype]=[]
        MRR[itype]=[]
for kkk in range(len(imgName)):# 2,000 tests
    start_time = timeit.default_timer()
    this_type = imgType[kkk]
    if this_type != 'clutter': # exclue clutter one
        query_src = feats[kkk] #imput img
        scores = np.dot(query_src, feats.T) #innner product
        rank_ID = np.argsort(scores)[::-1]# sort the result(return index)
        rank_score = scores[rank_ID]
        #img_list = [imgNames[index] for i,index in enumerate(rank_ID[0:Top_K])]
        rst_list = [imgType[index] for i,index in enumerate(rank_ID[0:Top_K])]
        PK,RK,F1K,MRRK = performance_evaluation(this_type,rst_list,Top_K)
        
        precision[this_type].append(PK)
        recall[this_type].append(RK)
        f1[this_type].append(F1K)
        MRR[this_type].append(MRRK)
        
        end_time = timeit.default_timer()
        time_elapsed.append(end_time - start_time)
    
global_average_runtime = np.mean(time_elapsed)
ave_p = []
ave_r=[]
ave_f=[]
ave_m=[]

def st(x):#format the float to 0.xxx
    return '{:.3f}'.format(x) #str.format("{0:<10.3f}", x)

##save results
file_name = 'eval_precision_'+str(Top_K)+'.txt'
with open(file_name,'w+') as f:
    print 'saving results...'
    for itype in np.unique(imgType):
        if itype != 'clutter':  # exclue clutter one
            pp = np.mean(precision[itype])
            rr = np.mean(recall[itype])
            ff = np.mean(f1[itype])
            mm = np.mean(MRR[itype])
            
            ave_p.append(np.mean(pp))
            ave_r.append(np.mean(rr))
            ave_f.append(np.mean(ff))
            ave_m.append(np.mean(mm))
        
            f.write(itype)
            f.write('\t')
            f.write(st(pp))
            f.write('\t')
            f.write(st(rr))
            f.write('\t')
            f.write(st(ff))
            f.write('\t')
            f.write(st(mm))
            f.write('\t')
            f.write('\n') 
     
    g_p=np.mean(ave_p)  
    g_r=np.mean(ave_r)   
    g_f=np.mean(ave_f)   
    g_m=np.mean(ave_m)    
    
    f.write('global_average_runtime    ')
    f.write(st(global_average_runtime))
    f.write('\n')
    f.write('global_precison    ')
    f.write(st(g_p))    
    f.write('\n')
    f.write('global_recall    ')
    f.write(st(g_r))    
    f.write('\n')
    f.write('global_f1    ')
    f.write(st(g_f))   
    f.write('\n') 
    f.write('global_MMR    ')
    f.write(st(g_m))  
    f.write('\n')   
    
print 'end of the programme'
    # global_average_evaluation_results 

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:56:36 2016

@author: Balakrishna
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:54:02 2016

@author: Balakrishna
"""

import glob
import numpy as np
#import struct
from scipy.io.wavfile import read
from sklearn import preprocessing
from lpc_all import lpc
#from lpc_all import j_all
wavs = []
mean_all = []
maxima_all = []
minima_all = []
rms_all = []
zcr_all = []
j_all = []

for filename in glob.glob('*.wav'):
    #print(filename)
    wavs.append(read(filename))
    signal = read(filename)
    signal2array= np.array(signal[1], dtype=float)  
    mean=np.mean(signal2array)
    mean_all.append(mean)
    maxima=signal2array.max()
    maxima_all.append(maxima)
    minima = signal2array.min()
    minima_all.append(minima)
    from numpy import mean, sqrt, square
    rms = sqrt(mean(square(signal2array)))
    rms_all.append(rms)
    zcr=(np.diff(np.sign(signal2array)) != 0).sum()
    #array_size=signal2array.size
    #zcr = count/array_size 
    zcr_all.append(zcr)


LPCcoeff,Err,Ka = lpc(signal2array,11)
for filename in glob.glob('*.wav'):
    #print(filename)
     wavs.append(read(filename))
     signal = read(filename)
     signal2array= np.array(signal[1], dtype=float)
     LPCcoeff,Err,Ka = lpc(signal2array,11)
     k=LPCcoeff[1:11]
     #j=k.tolist()
     j_all.append(k)
lpc_scaled= preprocessing.scale(j_all)
lpc_reshaped=lpc_scaled.reshape(10,30)
lpc_normalized = preprocessing.normalize(lpc_reshaped,norm='l2')
lpc_normalzed_cv = lpc_normalized.T
##j=LPCcoeff.tolist()
#lpc_all.append(LPCcoeff)
#b=list(lpc_all)
#l=b.tolist()
#LPCcoeff.tolist()

mean_scaled= preprocessing.scale(mean_all)
#q=mean_normalized.mean()
#w=X_scaled.std()
mean_reshaped=mean_scaled.reshape(1,-1)
#normalizer = preprocessing.Normalizer().fit(u)
mean_normalized = preprocessing.normalize(mean_reshaped,norm='l2')
mean_normalized_cv = mean_normalized.T


maxima_scaled= preprocessing.scale(maxima_all)
maxima_reshaped=maxima_scaled.reshape(1,-1)
maxima_normalized = preprocessing.normalize(maxima_reshaped,norm='l2')
maxima_normalized_cv = maxima_normalized.T

minima_scaled= preprocessing.scale(maxima_all)
minima_reshaped=maxima_scaled.reshape(1,-1)
minima_normalized = preprocessing.normalize(minima_reshaped,norm='l2')
minima_normalized_cv = minima_normalized.T

rms_scaled= preprocessing.scale(rms_all)
rms_reshaped=rms_scaled.reshape(1,-1)
rms_normalized = preprocessing.normalize(rms_reshaped,norm='l2')
rms_normalized_cv = rms_normalized.T

zcr_scaled= preprocessing.scale(zcr_all)
zcr_reshaped=rms_scaled.reshape(1,-1)
zcr_normalized = preprocessing.normalize(zcr_reshaped,norm='l2')
zcr_normalized_cv = zcr_normalized.T
#lpc_scaled= preprocessing.scale(lpc_all)
#lpc_reshaped=lpc_scaled.reshape(1,-1)
#lpc_normalized = preprocessing.normalize(lpc_reshaped,norm='l2')
#list_all= list(zip(mean_normalized,maxima_normalized,minima_normalized,rms_normalized,zcr_normalized))
#y=list_all.tolist()
f_matrix=np.concatenate((maxima_normalized_cv,minima_normalized_cv,mean_normalized_cv,rms_normalized_cv,zcr_normalized_cv),axis=1)
final_matrix=np.concatenate((f_matrix,lpc_normalzed_cv),axis=1)
#z= list(chain(maxima_normalized_cv,minima_normalized_cv,rms_normalized_cv))
X = np.array([final_matrix[0,:], final_matrix[1,:],final_matrix[2,:],
              final_matrix[3,:],final_matrix[4,:],final_matrix[5,:],
              final_matrix[6,:],final_matrix[7,:],final_matrix[8,:],
              final_matrix[9,:],final_matrix[10,:],final_matrix[11,:],
              final_matrix[12,:],final_matrix[13,:],final_matrix[14,:],
              final_matrix[15,:],final_matrix[16,:],final_matrix[17,:],
              final_matrix[18,:],final_matrix[19,:],final_matrix[20,:],
              final_matrix[21,:],final_matrix[22,:],final_matrix[23,:],
              final_matrix[24,:],final_matrix[25,:],final_matrix[26,:],
              final_matrix[27,:],final_matrix[28,:],final_matrix[29,:]])
y = np.array([1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3])
from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print(clf.predict([-2.450731070511635856e-01,-2.450731070511635856e-01,
                   8.721489461596376724e-02,
                   -4.069402792164062976e-01	,-4.069402792164062976e-01]))
 
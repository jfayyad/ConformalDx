import torch
import pickle
import numpy as np
from RAPS import RAPS

from misclassify_baseline import test

qhat, reg_vec = RAPS(1000,0.1)
correct, missclasify = test()

u_corr = []
u_wrong = []
disallow_zero_sets = False # Set this to False in order to see the coverage upper bound hold
rand = False

for i in range(len(correct)):
    _smx = np.array(correct[i][2].cpu())
    _pi = np.argsort(_smx)[::-1]
    _srt = np.take_along_axis(_smx,_pi,axis=0)
    _srt_reg = _srt + reg_vec.squeeze()
    _srt_reg_cumsum = _srt_reg.cumsum()
    _ind = (_srt_reg_cumsum - np.random.rand()*_srt_reg) <= qhat if rand else _srt_reg_cumsum - _srt_reg <= qhat
    if disallow_zero_sets: _ind[0] = True
    prediction_set = np.take_along_axis(_ind,_pi.argsort(),axis=0)
    u_corr.append((np.nonzero(prediction_set)[0].size)/7)

for i in range(len(missclasify)):
    _smx = np.array(missclasify[i][2].cpu())
    _pi = np.argsort(_smx)[::-1]
    _srt = np.take_along_axis(_smx,_pi,axis=0)
    _srt_reg = _srt + reg_vec.squeeze()
    _srt_reg_cumsum = _srt_reg.cumsum()
    _ind = (_srt_reg_cumsum - np.random.rand()*_srt_reg) <= qhat if rand else _srt_reg_cumsum - _srt_reg <= qhat
    if disallow_zero_sets: _ind[0] = True
    prediction_set_w = np.take_along_axis(_ind,_pi.argsort(),axis=0)
    u_wrong.append((np.nonzero(prediction_set_w)[0].size)/7)

correct_array = np.array(u_corr)
wrong_array = np.array(u_wrong)


np.save("RAPS_Conformal_correct.npy",correct_array)
np.save("RAPS_Conformal_wrong.npy",wrong_array)


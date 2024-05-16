import numpy as np
import torch

torch.manual_seed(123)


def RAPS(n, alpha):

    smax = np.load("/home/jfayyad/PycharmProjects/Conformal/smxNOV6.npz")
    lbl = np.load("/home/jfayyad/PycharmProjects/Conformal/labelsNOV6.npz")

    smx = smax["arr_0"]
    labels = lbl["arr_0"].squeeze().astype(int)

    lam_reg = 0.9
    k_reg = 3
    disallow_zero_sets = False # Set this to False in order to see the coverage upper bound hold
    rand = True # Set this to True in order to see the coverage upper bound hold
    reg_vec = np.array(k_reg*[0,] + (smx.shape[1]-k_reg)*[lam_reg,])[None,:]

    idx = np.array([1] * n + [0] * (smx.shape[0]-n)) > 0
    np.random.shuffle(idx)
    cal_smx, val_smx = smx[idx,:], smx[~idx,:]
    cal_labels, val_labels = labels[idx], labels[~idx]


    cal_pi = cal_smx.argsort(1)[:,::-1]
    cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]



    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg_cumsum- np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)

    empirical_coverage = prediction_sets[np.arange(n_val),val_labels].mean()

    print(f"The empirical coverage is: {empirical_coverage}")
    print(f"The quantile is: {qhat}")

    return qhat , reg_vec

# RAPS(1000, 0.1)

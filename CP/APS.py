import numpy as np
import torch

torch.manual_seed(123)


def APS(n, alpha):
    smax = np.load("/home/jfayyad/PycharmProjects/Conformal/smxRes50.npz")
    lbl = np.load("/home/jfayyad/PycharmProjects/Conformal/labelsRes50.npz")

    smx = smax["arr_0"]
    labels = lbl["arr_0"].squeeze().astype(int)

    # Split the softmax scores into calibration and validation sets (save the shuffling)
    idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
    np.random.shuffle(idx)
    cal_smx, val_smx = smx[idx, :], smx[~idx, :]
    cal_labels, val_labels = labels[idx], labels[~idx]


    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]

    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )

    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)

    empirical_coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]), val_labels
    ].mean()
    print(f"The empirical coverage is: {empirical_coverage}")

    return qhat



# i = 20
#
# test_image = test_data[i][0].unsqueeze(dim=0).to(device)
# test_label = test_data[i][1]
# smx_test = torch.nn.functional.softmax(model(test_image))
# prediction_set = smx_test > 1-qhat
# prediction_set = prediction_set.cpu().numpy().flatten()
# label_strings=np.array(["0","1","2","3","4","5","6"])
#
# print(f"The prediction set is: {list(label_strings[prediction_set])}")
#
# print("The correct label is: ",test_label)

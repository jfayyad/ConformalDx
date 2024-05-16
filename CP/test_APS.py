import torch
import pickle
import numpy as np
from APS import APS
from misclassify_baseline import test
import matplotlib.pyplot as plt

save = 0
size = 1000

# Loop over different values of alpha
for alpha in np.arange(0.1, 0.2, 0.1):
    qhat = APS(size, alpha)

    correct, missclassify = test()

    u_corr = []
    u_wrong = []

    correct_highU = []
    correct_lowU = []

    wrong_highU = []
    wrong_lowU =[]

    for i in range(len(correct)):
        smx = correct[i][2]
        img_pi = smx.argsort(descending=True)
        img_srt = torch.gather(smx, 0, img_pi).cumsum(dim=0)
        prediction_set = torch.take(img_pi.argsort(), torch.nonzero((img_srt <= qhat)))
        u1 = (prediction_set.size()[0] / 7)
        img_tuple = (correct[i][0], correct[i][3])  # (image, image_name)
        if u1 > 0.6:
            correct_highU.append(img_tuple)
        elif u1 < 0.3:
            correct_lowU.append(img_tuple)
        u_corr.append(u1)

    for i in range(len(missclassify)):
        smx = missclassify[i][2]
        img_pi = smx.argsort(descending=True)
        img_srt = torch.gather(smx, 0, img_pi).cumsum(dim=0)
        prediction_set = torch.take(img_pi.argsort(), torch.nonzero((img_srt <= qhat)))

        u2 = (prediction_set.size()[0] / 7)
        img_tuple = (missclassify[i][0], missclassify[i][3])  # (image, image_name)
        if u2 > 0.6:
            wrong_highU.append(img_tuple)
        elif u2 <= 0.4:
            wrong_lowU.append(img_tuple)
        u_wrong.append(u2)

    correct_array = np.array(u_corr)
    wrong_array = np.array(u_wrong)

    # Convert alpha to corresponding alphabet letter
    alphabet_letter = chr(ord('a') + int(alpha * 10 - 1))

    # Save files with alphabet letter in the name
    if save:
        np.save(f"{alphabet_letter}_APS_Conformal_VGG2_Correct_alpha_{alpha}_size_{size}.npy", correct_array)
        np.save(f"{alphabet_letter}_APS_Conformal_VGG2_Wrong_alpha_{alpha}_size_{size}.npy", wrong_array)

def display_image(image):
    # If the image has three channels (RGB)
    if image.shape[0] == 3:
        # Transpose the image from [3, 224, 224] to [224, 224, 3] for Matplotlib
        image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


correct_highU_indices = np.random.choice(len(correct_highU), size=5, replace=False)
correct_lowU_indices = np.random.choice(len(correct_lowU), size=5, replace=False)

wrong_highU_indices = np.random.choice(len(wrong_highU), size=5, replace=False)
wrong_lowU_indices = np.random.choice(len(wrong_lowU), size=5, replace=False)


correct_h = [correct_highU[i] for i in correct_highU_indices]
correct_l = [correct_lowU[i] for i in correct_lowU_indices]

wrong_h = [wrong_highU[i] for i in wrong_highU_indices]
wrong_l = [wrong_lowU[i] for i in wrong_lowU_indices]

correct_h_filename = 'correct_h.pkl'
correct_l_filename = 'correct_l.pkl'
wrong_h_filename = 'wrong_h.pkl'
wrong_l_filename = 'wrong_l.pkl'

# Save lists using pickle
with open(correct_h_filename, 'wb') as f:
    pickle.dump(correct_h, f)

with open(correct_l_filename, 'wb') as f:
    pickle.dump(correct_l, f)

with open(wrong_h_filename, 'wb') as f:
    pickle.dump(wrong_h, f)

with open(wrong_l_filename, 'wb') as f:
    pickle.dump(wrong_l, f)

















# import torch
# import pickle
# import numpy as np
# from APS import APS
#
# from misclassify_baseline import test
#
#
# size = 1000
# alpha = 1.0
#
# qhat = APS(size,alpha)
#
# correct, missclasify = test()
#
#
# u_corr = []
# u_wrong = []
# for i in range(len(correct)):
#     smx = correct[i][2]
#     labels = correct[i][1]
#     img_pi = smx.argsort(descending=True)
#     img_srt = torch.gather(smx, 0, img_pi).cumsum(dim=0)
#     prediction_set = torch.take(img_pi.argsort(), torch.nonzero((img_srt <= qhat)))
#     u_corr.append((prediction_set.size()[0]/7))
#
#
# for i in range(len(missclasify)):
#     smx = missclasify[i][2]
#     labels = missclasify[i][1]
#     img_pi = smx.argsort(descending=True)
#     img_srt = torch.gather(smx, 0, img_pi).cumsum(dim=0)
#     prediction_set = torch.take(img_pi.argsort(), torch.nonzero((img_srt <= qhat)))
#     u_wrong.append((prediction_set.size()[0]/7))
#
# correct_array = np.array(u_corr)
# wrong_array = np.array(u_wrong)
#
# np.save(f"APS_Conformal_Correct_alpha_{alpha}_size_{size}.npy",correct_array)
# np.save(f"APS_Conformal_Wrong_alpha_{alpha}_size_{size}.npy",wrong_array)

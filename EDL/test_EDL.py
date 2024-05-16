import torch
import Resnet
from helpers import get_device
from dataloaders import load_data
from losses import relu_evidence
import numpy as np
from torchvision import transforms, datasets, models



# arch = getattr(Resnet, 'resnet50')
# model = arch(pretrained=True, dropout_rate=0.5)
# if hasattr(model, 'fc'):
#     model.fc = torch.nn.Linear(model.fc.in_features, 7)
# device = get_device()
# saved_weights_path = '/home/jfayyad/PycharmProjects/Conformal/EDL/results/model_uncertainty_mse_Resnet50.pt'
# checkpoint =torch.load(saved_weights_path)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.to(device)

model = models.vgg11(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 7)

saved_weights_path = '/home/jfayyad/PycharmProjects/Conformal/EDL/results/model_uncertainty_mse_VGG.pt'  # Replace with the actual path to your saved weights
checkpoint =torch.load(saved_weights_path)
model.load_state_dict(checkpoint["model_state_dict"])
device = get_device()
model = model.to(device)



_,_,test_loader = load_data()



def testing_edl():
    num_classes =7
    misclassified_samples_total = []
    correct_samples_total = []
    correct = 0
    total = 0
    for i, (image, label) in enumerate(test_loader):
        model.eval()
        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            _, preds = torch.max(output, 1)
            total += label.size(0)
            correct += (preds == label).sum().item()
            misclassified_indices = (preds != label).nonzero().squeeze()
            correct_indices = (preds == label).nonzero().squeeze()

            if len(misclassified_indices.size()) > 0:
                for index in misclassified_indices:
                    misclassified_samples_total.append((image[index], label[index], output[index], uncertainty[index]))

            for index in correct_indices:
                    correct_samples_total.append((image[index], label[index], output[index], uncertainty[index]))

    acc = 100 * correct / total
    print("The accuracy is ", acc)

    return misclassified_samples_total, correct_samples_total


u_corr = []
u_wrong = []

missclasify, correct = testing_edl()

for i in range(len(correct)):
    uncertainty = correct[i][3]
    u_corr.append(uncertainty)


for i in range(len(missclasify)):
    uncertainty_w = missclasify[i][3]
    u_wrong.append(uncertainty_w)


correct_array = [tensor.cpu().numpy() for tensor in u_corr]
wrong_array = [tensor.cpu().numpy() for tensor in u_wrong]

correct_array = np.concatenate(correct_array)
wrong_array = np.concatenate(wrong_array)


np.save("EDL_correct_VGG.npy",correct_array)
np.save("EDL_wrong_VGG.npy",wrong_array)

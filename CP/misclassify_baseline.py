import pickle
import torch

import Networks.Resnet
from EDL.helpers import get_device
from data_handler.dataloaders import load_data
from torchvision import transforms, datasets, models


def test():

    arch = getattr(Networks.Resnet, 'resnet50')
    model = arch(pretrained=True, dropout_rate=0.5)
    if hasattr(model, 'fc'):
        model.fc = torch.nn.Linear(model.fc.in_features, 7)
    device = get_device()

    saved_weights_path = '/home/jfayyad/PycharmProjects/Conformal/EDL/results/ResNet50.pt'  # Replace with the actual path to your saved weights
    checkpoint =torch.load(saved_weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    _,_,test_loader = load_data()
    misclassified_samples_total = []
    correct_samples_total = []
    correct = 0
    total = 0

    for i, (image, label, image_name) in enumerate(test_loader):
        model.eval()
        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)

            out = torch.nn.functional.softmax(model(image))
            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()

            misclassified_indices = (pred != label).nonzero().squeeze()

            correct_indices = (pred == label).nonzero().squeeze()

            if len(misclassified_indices.size()) > 0:
                for index in misclassified_indices:
                    misclassified_samples_total.append((image[index], label[index], out[index], image_name[index]))

            for index in correct_indices:
                correct_samples_total.append((image[index], label[index], out[index], image_name[index]))

    acc = 100 * correct / total
    print("The accuracy is ", acc)
    return correct_samples_total, misclassified_samples_total

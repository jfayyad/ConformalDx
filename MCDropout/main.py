import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from utils import get_device
from data_handler.dataloaders import load_data
from tqdm import tqdm
import Networks.Resnet
from arguments import get_args

args = get_args()

torch.manual_seed(args.seed)
device = get_device()


arch = getattr(Networks.Resnet, 'resnet50')
model = arch(pretrained=True, dropout_rate=0.5)
if hasattr(model, 'fc'):
    model.fc = torch.nn.Linear(model.fc.in_features, 7)
saved_weights_path = args.save_dir
checkpoint =torch.load(saved_weights_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

class ResNetWithDropout(nn.Module):
    def __init__(self, original_model, dropout_prob):
        super(ResNetWithDropout, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.dropout= original_model.dropout
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.dropout_last = nn.Dropout(dropout_prob)
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.dropout_last(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

MCD_model = ResNetWithDropout(model,0.5)

_, _, dataloader_test = load_data()

def calculate_metrics(predictions, labels):
    """ Function to calculate classification metrics """
    predicted_labels = np.argmax(predictions, axis=1)
    correct = np.sum(predicted_labels == labels)
    total = len(labels)
    accuracy = correct / total

    return accuracy

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_monte_carlo_predictions(data_loader,
                                forward_passes,
                                model,
                                n_classes,
                                n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)

    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for i, (image, label) in enumerate(data_loader):
            image = image.to(torch.device('cuda'))
            with torch.no_grad():
                output = model(image)
                output = softmax(output)  # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                           axis=-1), axis=0)  # shape (n_samples,)
    return mean, variance

all_labels = np.empty(0, dtype=int)
for i, (images, labels) in enumerate(dataloader_test):
    all_labels = np.concatenate((all_labels, labels.numpy()))

m,v = get_monte_carlo_predictions(dataloader_test,1000,MCD_model,7,len(dataloader_test.dataset))


accuracy = calculate_metrics(m,all_labels)

predicted_label = np.argmax(m, axis=1)
predicted_variances = v[np.arange(len(predicted_label)), predicted_label]

correct_idx = np.where(predicted_label == all_labels)
wrong_idx = np.where(predicted_label != all_labels)

correct_uncertainty = predicted_variances[correct_idx]
missclassified_uncertainty = predicted_variances[wrong_idx]

np.save("DO_correct_res50.npy",correct_uncertainty)
np.save("DO_wrong.npy_res50",missclassified_uncertainty)

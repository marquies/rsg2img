import torch
import os
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from PIL import Image
from torchvision import models, transforms

from torchvision.models.inception import inception_v3
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.linalg import sqrtm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#class CustomImageDataset(Dataset):
#    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#        self.img_labels = pd.read_csv(annotations_file)
#        self.img_dir = img_dir
#        self.transform = transform
#        self.target_transform = target_transform
#
#    def __len__(self):
#        return len(self.img_labels)
#
#    def __getitem__(self, idx):
#        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#        image = Image.open(img_path).convert("RGB")
#        label = self.img_labels.iloc[idx, 1]
#        if self.transform:
#            image = self.transform(image)
#        if self.target_transform:
#            label = self.target_transform(label)
#        return image, label



class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor
        #dtype = torch.mps.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    #mps_device = torch.device("mps")
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    #inception_model.to(mps_device)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def main():

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    #cifar = dset.CIFAR10(root='data/', download=True,
    #                         transform=transforms.Compose([
    #                             transforms.Resize(32),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                         ])
    #)

    #dict = unpickle('data/cifar-10-batches-py/data_batch_1')
    #print(dict.keys())
    #print(dict[b'filenames'])
    ## Define transformations
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    src = 'data/metaverse/original'
    gen = 'data/metaverse/generated'

    print('Total images in src_path:', len(next(os.walk(src))[2]))
    print('Total images in gen_path:', len(next(os.walk(gen))[2]))

    # Create datasets
    original_dataset = CustomImageDataset(img_dir=src, transform=transform)
    generated_dataset = CustomImageDataset(img_dir=gen, transform=transform)
    
    # Create dataloaders
    original_loader = DataLoader(original_dataset, batch_size=32, shuffle=False, num_workers=4)
    generated_loader = DataLoader(generated_dataset, batch_size=32, shuffle=False, num_workers=4)
    dtype = torch.FloatTensor
    # Load pre-trained Inception v3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()

    #IgnoreLabelDataset(cifar)

    #print ("Calculating Inception Score...")
    #print (inception_score(IgnoreLabelDataset(cifar), cuda=False, batch_size=32, resize=True, splits=10))


    def get_predictions(loader):
        preds = []
        for batch in loader:
            with torch.no_grad():
                batch = batch.cpu()
                pred = inception_model(batch)
                preds.append(F.softmax(pred, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def inception_score(preds, splits=10):
        scores = []
        N = preds.shape[0]
        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores.append(np.exp(entropy(py, np.mean(part, axis=0))))
        return np.mean(scores), np.std(scores)

    # Calculate predictions
    original_preds = get_predictions(original_loader)
    generated_preds = get_predictions(generated_loader)

    # Calculate Inception Scores
    #original_score, original_std = inception_score(original_preds)
    #generated_score, generated_std = inception_score(generated_preds)

    #print(f'Original predictions shape: {original_preds.shape}')
    #print(f'Generated predictions shape: {generated_preds.shape}')
    #print(f'Original predictions sample: {original_preds[:5]}')
    #print(f'Generated predictions sample: {generated_preds[:5]}')

    # Check for NaN values in predictions
    if np.any(np.isnan(original_preds)) or np.any(np.isnan(generated_preds)):
        print("NaN values found in predictions")
    else:
        # Calculate Inception Scores
        original_score, original_std = inception_score(original_preds)
        generated_score, generated_std = inception_score(generated_preds)

        print(f'Original Inception Score: {original_score} ± {original_std}')
        print(f'Generated Inception Score: {generated_score} ± {generated_std}')

    return

    def get_features(loader):
        features = []
        for batch in loader:
            with torch.no_grad():
                batch = batch.cpu()
                pred = inception_model(batch)[0]  # Get the features from the model
                features.append(pred.cpu().numpy())
        return np.concatenate(features, axis=0)

    # Calculate features
    original_features = get_features(original_loader)
    generated_features = get_features(generated_loader)

    def calculate_fid(original_features, generated_features):
        mu1 = np.mean(original_features, axis=0)
        mu2 = np.mean(generated_features, axis=0)
        sigma1 = np.cov(original_features, rowvar=False)
        sigma2 = np.cov(generated_features, rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    # Calculate FID
    fid_value = calculate_fid(original_features, generated_features)
    print(f'Fréchet Inception Distance (FID): {fid_value}')

if __name__ == "__main__":
	main()

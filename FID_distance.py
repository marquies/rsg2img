#!/usr/bin/env python
# coding: utf-8

# # <center style='color:deeppink'>Calculate FID (`Frechet Inception Distance`) using PyTorch</center>

# # 1. Import the required libraries

# In[2]:


import torch
print('PyTorch version:', torch.__version__)
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
print('Torchvision version:', torchvision.__version__)
from torchvision import transforms

import os
from PIL import Image
import numpy as np
from scipy import linalg
import pathlib
from tqdm import tqdm

# custom module
from INCEPTION import InceptionV3


# # 2. Define `ImagePathDataset` class

# In[3]:


IMAGE_EXTENSIONS = {'jpg'}


# In[4]:


class ImagePathDataset(Dataset):
    def __init__(self, files, transform=None):
        
        self.files = files
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


# # 3. Define `get_activations`

# In[5]:


def get_activations(files, model, batch_size, dims, device='cpu'):
    
    model.eval()
    
    if batch_size > len(files):
        batch_size = len(files)
        
    dataset = ImagePathDataset(files, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    pred_arr = np.empty((len(files), dims))
    start_idx = 0
    
    for batch in tqdm(data_loader):
        batch = batch.to(device)
        
        with torch.inference_mode():
            pred = model(batch)[0]
            
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx+pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
        
    return pred_arr


# # 4. Define `calculate_frechet_distance`

# In[6]:


def calculate_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces sigular product; adding %s to diagonal cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# # 5. Define `calculate_activation_statistics`

# In[7]:


def calculate_activation_statistics(files, model, batch_size, dims, device='cpu'):
    
    act = get_activations(files, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    
    return mu, sigma


# # 6. Define `compute_statistics_of_path`

# In[8]:


def compute_statistics_of_path(path, model, batch_size, dims, device='cpu'):
    
    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))])
    mu, sigma = calculate_activation_statistics(files, model, batch_size, dims, device)
        
    return mu, sigma


# # 7. Calculate `FID distance`

# In[9]:


def calculate_fid_given_paths(path1, path2, batch_size, dims, device='cpu'):
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    mu1, sigma1 = compute_statistics_of_path(path1, model, batch_size, dims, device)
    mu2, sigma2 = compute_statistics_of_path(path2, model, batch_size, dims, device)
    
    fid_value = calculate_frechet_distance(mu1, mu2, sigma1, sigma2)
    return print('FID distance:', round(fid_value, 3))


# In[16]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 50
dims = 2048

src_path = 'data/metaverse-sd3med/original'
gen_path = 'data/metaverse-sd3med/generated'

print('Total images in src_path:', len(next(os.walk(src_path))[2]))
print('Total images in gen_path:', len(next(os.walk(gen_path))[2]))



# In[17]:


calculate_fid_given_paths(path1=src_path, path2=gen_path, batch_size=batch_size, dims=dims, device=device)


# # A lower value of `FID distance` denotes more similarity with source images

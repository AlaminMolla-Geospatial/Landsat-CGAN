#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import rasterio
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# 1. Directories and Config
# -----------------------------
INPUT_DIR = '/home/amolla/DNN/DL_Fall25/FinalProject_EEE598/LandsatData/combined_geotiffs'
OUTPUT_DIR = '/home/amolla/DNN/DL_Fall25/FinalProject_EEE598/LandsatData/landsatCGAN'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    'in_bands':[1,2,3,4,5],
    'out_band':6,
    'patch_size':256,
    'patches_per_image':50,
    'batch_size':2,
    'epochs':50,
    'device':'cuda' if torch.cuda.is_available() else 'cpu',
    'lambda_l1':100.0,
    'lr':2e-4,
    'n_visual_patches':10
}

# -----------------------------
# 2. Dataset
# -----------------------------
class PairedPatchDataset(Dataset):
    def __init__(self, pairs, in_bands, out_band, patch_size=256, patches_per_image=50):
        self.pairs = pairs
        self.in_bands = in_bands
        self.out_band = out_band
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

    def __len__(self):
        return len(self.pairs) * self.patches_per_image

    def _read(self, path, bands):
        with rasterio.open(path) as src:
            if isinstance(bands,(list,tuple)):
                arr = np.stack([src.read(b) for b in bands],axis=0).astype(np.float32)
            else:
                arr = src.read(bands).astype(np.float32)[None,...]
        return arr

    def __getitem__(self, idx):
        file_idx = (idx // self.patches_per_image) % len(self.pairs)
        in_path, out_path = self.pairs[file_idx]
        inp = self._read(in_path,self.in_bands)
        tgt = self._read(out_path,self.out_band)
        if tgt.ndim==2: tgt = tgt[None,...]

        C,H,W = inp.shape
        ps = self.patch_size
        y0 = random.randint(0,H-ps)
        x0 = random.randint(0,W-ps)
        in_patch = inp[:,y0:y0+ps,x0:x0+ps]
        tgt_patch = tgt[:,y0:y0+ps,x0:x0+ps]

        # Normalize L1 input bands to [-1,1]
        def norm(a):
            lo, hi = np.percentile(a, 2), np.percentile(a,98)
            a = np.clip((a-lo)/(hi-lo+1e-6),0,1)
            return a*2-1
        in_patch_norm = norm(in_patch)

        # Normalize L2 thermal and store lo/hi
        lo, hi = np.percentile(tgt_patch,2), np.percentile(tgt_patch,98)
        tgt_patch_norm = np.clip((tgt_patch-lo)/(hi-lo+1e-6),0,1)*2-1

        return (torch.tensor(in_patch_norm,dtype=torch.float32),
                torch.tensor(tgt_patch_norm,dtype=torch.float32),
                lo, hi)

# -----------------------------
# 3. Model Definitions
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.conv(x)

class UNetGen(nn.Module):
    def __init__(self,in_ch=5,out_ch=1,base=64):
        super().__init__()
        self.enc1 = DoubleConv(in_ch,base)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2),DoubleConv(base,base*2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2),DoubleConv(base*2,base*4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2),DoubleConv(base*4,base*8))
        self.mid = DoubleConv(base*8,base*8)
        self.dec4 = DoubleConv(base*8 + base*8, base*4)
        self.dec3 = DoubleConv(base*4 + base*4, base*2)
        self.dec2 = DoubleConv(base*2 + base*2, base)
        self.final = nn.Sequential(nn.Conv2d(base,out_ch,1), nn.Tanh())

    def forward(self,x):
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        m = self.mid(c4)
        u4 = F.interpolate(m,size=c4.shape[2:],mode='bilinear',align_corners=False)
        d4 = self.dec4(torch.cat([u4,c4],dim=1))
        u3 = F.interpolate(d4,size=c3.shape[2:],mode='bilinear',align_corners=False)
        d3 = self.dec3(torch.cat([u3,c3],dim=1))
        u2 = F.interpolate(d3,size=c2.shape[2:],mode='bilinear',align_corners=False)
        d2 = self.dec2(torch.cat([u2,c2],dim=1))
        u1 = F.interpolate(d2,size=c1.shape[2:],mode='bilinear',align_corners=False)
        out = self.final(u1)
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self,in_ch=6,base=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch,base,4,2,1), nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(base,base*2,4,2,1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(base*2,base*4,4,2,1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(base*4,1,4,1,1)
        )
    def forward(self,x): return self.model(x)

# -----------------------------
# 4. Detect all L1-L2 pairs 
# -----------------------------
all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')])
l1_files = [f for f in all_files if 'L1TP' in f]
l2_files = [f for f in all_files if 'L2SP' in f]

def l1_key(filename):
    parts = filename.split('_')
    return f"{parts[0]}_{parts[2]}_{parts[3]}"

def l2_key(filename):
    parts = filename.split('_')
    return f"{parts[0]}_{parts[2]}_{parts[3]}"

l2_dict = {l2_key(f): f for f in l2_files}

pairs = []
for l1 in l1_files:
    key = l1_key(l1)
    if key in l2_dict:
        pairs.append((os.path.join(INPUT_DIR,l1), os.path.join(INPUT_DIR,l2_dict[key])))

print(f"Total L1-L2 pairs found: {len(pairs)}")

# -----------------------------
# 5. Leave-one-out split
# -----------------------------
test_idx = 0  # choose which pair to test
train_pairs = [p for i,p in enumerate(pairs) if i != test_idx]
test_pair = [pairs[test_idx]]

# -----------------------------
# 6. Training setup
# -----------------------------
device = CONFIG['device']
G = UNetGen(in_ch=len(CONFIG['in_bands']),out_ch=1).to(device)
D = PatchDiscriminator(in_ch=len(CONFIG['in_bands'])+1).to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=CONFIG['lr'], betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=CONFIG['lr'], betas=(0.5,0.999))

l1_loss = nn.L1Loss()
adv_loss = nn.BCEWithLogitsLoss()

G_losses = []
D_losses = []

# -----------------------------
# 7. Training loop
# -----------------------------
train_dataset = PairedPatchDataset(train_pairs, CONFIG['in_bands'], CONFIG['out_band'],
                                   patch_size=CONFIG['patch_size'], patches_per_image=CONFIG['patches_per_image'])
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

for epoch in range(CONFIG['epochs']):
    G.train(); D.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    for xb, yb, lo, hi in loop:
        xb, yb = xb.to(device), yb.to(device)

        # Train Discriminator
        with torch.no_grad():
            fake = G(xb)
        D_real = D(torch.cat([xb,yb],dim=1))
        D_fake = D(torch.cat([xb,fake],dim=1))
        real_label = torch.ones_like(D_real)
        fake_label = torch.zeros_like(D_fake)
        loss_D = (adv_loss(D_real,real_label)+adv_loss(D_fake,fake_label))*0.5
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # Train Generator
        fake = G(xb)
        D_fake_for_G = D(torch.cat([xb,fake],dim=1))
        loss_G = adv_loss(D_fake_for_G, real_label) + CONFIG['lambda_l1']*l1_loss(fake,yb)
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        loop.set_postfix({'loss_D': float(loss_D.item()), 'loss_G': float(loss_G.item())})
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

# -----------------------------
# 8. Evaluation on test pair
# -----------------------------
G.eval()
all_metrics = []

test_dataset = PairedPatchDataset(test_pair, CONFIG['in_bands'], CONFIG['out_band'],
                                  patch_size=CONFIG['patch_size'], patches_per_image=CONFIG['patches_per_image'])

indices = random.sample(range(len(test_dataset)), CONFIG['n_visual_patches'])
for i, idx in enumerate(indices):
    xb_patch, yb_patch, lo_patch, hi_patch = test_dataset[idx]
    xb_patch = xb_patch.unsqueeze(0).to(device)
    with torch.no_grad(): pred_norm = G(xb_patch).cpu().numpy()[0,0]
    true_norm = yb_patch[0].numpy()
    lo_val, hi_val = lo_patch, hi_patch

    # Denormalize to Kelvin
    pred_real = (pred_norm+1)/2*(hi_val-lo_val)+lo_val
    true_real = (true_norm+1)/2*(hi_val-lo_val)+lo_val

    # Metrics
    mae = np.mean(np.abs(pred_real-true_real))
    rmse = np.sqrt(np.mean((pred_real-true_real)**2))
    bias = np.mean(pred_real-true_real)
    ssim_val = ssim(pred_real, true_real, data_range=true_real.max()-true_real.min())
    all_metrics.append([f'test_patch{i+1}', mae, rmse, bias, ssim_val])

    # L1 RGB visualization
    l1_rgb = xb_patch[0].cpu().numpy()[[2,1,0]]  # B4,B3,B2
    l1_rgb = (l1_rgb - l1_rgb.min())/(l1_rgb.max()-l1_rgb.min()+1e-6)
    l1_rgb = np.transpose(l1_rgb,(1,2,0))

    # Plot side by side
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(l1_rgb)
    plt.title("L1 RGB Input")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(true_real,cmap='inferno')
    plt.title("True Thermal (K)")
    plt.colorbar(label='Temperature (K)')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pred_real,cmap='inferno')
    plt.title(f"Predicted Thermal (K)\nMAE:{mae:.2f} RMSE:{rmse:.2f}\nBias:{bias:.2f} SSIM:{ssim_val:.2f}")
    plt.colorbar(label='Temperature (K)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,f'test_patch{i+1}_visualization.png'), dpi=300)
    plt.close()

# -----------------------------
# 9. Save metrics CSV
# -----------------------------
csv_file = os.path.join(OUTPUT_DIR,'test_pair_patch_metrics.csv')
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Patch','MAE(K)','RMSE(K)','Bias(K)','SSIM'])
    writer.writerows(all_metrics)

# -----------------------------
# 10. Save Loss Curves
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(G_losses,label='Generator Loss')
plt.plot(D_losses,label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('GAN Training Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'loss_curves.png'), dpi=300)
plt.close()

print("Training, testing, visualization, metrics, and loss curve saving complete.")

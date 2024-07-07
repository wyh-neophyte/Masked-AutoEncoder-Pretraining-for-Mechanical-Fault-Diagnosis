import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from data.normaldataset import *
from conf import *
from models.model.transformer import Transformer
import torch
from torch import nn
import torch.nn.functional as F
import time
import argparse

parser = argparse.ArgumentParser(description='train transformer')
parser.add_argument('--mae', type=bool, default=True)
parser.add_argument('--mae_pretrained', type=str, 
                    default='/root/MAE/snapshot/checkpoint_e50.pth'
                    #'/root/transformer2/snapshot-reconstructionloss/checkpoint_e25.pth'
                    #'/root/transformer2/snapshot/checkpoint_e25.pth')
                   )
args = parser.parse_args()

def extractfeatures(model, dataloader):
    model.eval()
    featurelist, labellist = [], []
    count = 0
    for signal, label in tqdm(dataloader):
        if count>30:
            break
        signal, label = signal.cuda().float(), label.cuda().long()
        _, features = model(signal, returnfeature=True)
        featurelist.append(features.detach().cpu())
        labellist.append(label.detach().cpu())
        count += 1
    print('feature', featurelist[0].shape)
    print('label', labellist[0].shape)
    return torch.cat(featurelist, dim=0), torch.cat(labellist)

def tsneplot(features, labels, savename):
    # 将PyTorch张量转换为NumPy数组
    features_np = features.numpy()
    labels_np = labels.numpy()

    # 使用t-SNE将特征降维到二维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_np)
    print(f'features_2d:{features_2d.shape}')
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels_np):
        indices = (labels_np == label)
        print(f"indices:{indices.shape}")
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'Class {label}', alpha=0.6)

    plt.legend()
    plt.title("t-SNE of Sample Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(savename)
    
    
def main():
    datasetpath = '/root/autodl-tmp/LinGang'
    wtlevel=2
    # train dataset and dataloader
    traindataset = LinGangTestDataset(datasetpath, speed=['1800'], datatype='train', ratio=0.8, wtlevel=wtlevel)
    trainDataloader = DataLoader(traindataset, batch_size=60, shuffle=True)

    # test dataset and dataloader
    testdataset = LinGangTestDataset(datasetpath, speed=['1200'], wtlevel=wtlevel)
    testDataloader = DataLoader(testdataset, batch_size=60, shuffle=True)
    
    patchnum = 64
    # transformer model
    model = Transformer(in_channels=wtlevel+1,
                        d_model=d_model,
                        max_len=max_len,
                        patchnum=patchnum,
                        ffn_hidden=ffn_hidden,
                        n_head=n_heads,
                        n_layers=n_layers,
                        drop_prob=drop_prob,
                        device=device,
                        classnum=5).to(device)
    
    if args.mae:
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.mae_pretrained)['state_dict'], strict=False)
        print("missing keys", missing_keys)
        savename = 'mae-pretrained-tsne-plot.jpg'
    else:
        savename = 'wo-mae-tsne-plot.jpg'
        
    features, labels = extractfeatures(model, testDataloader)
    print('concat feature into', features.shape)
    print('concat label into', labels.shape)
    tsneplot(features, labels, savename=savename)
    
main()
import os.path
import sys

import torch
from tqdm import tqdm
import time
from data.maedataset import *
from conf import *
from models.model.transformer import Transformer
from train.visualize_mae import *
from train.masktokens import *
from torch import nn, optim
from torch.optim import Adam, AdamW
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

    
def train(model, dataloader, optimizer, weight=[1, 1, 1], patchnum=32, mask_ratio=0.50):
    totalloss = [0, 0, 0]
    for signal, label in tqdm(dataloader):
        model.train()
        masktoken = masktokens(patchnum=patchnum, mask_ratio=mask_ratio)
        signal, label, masktoken = signal.cuda().float(), label.cuda().float(), masktoken.cuda().long()
        outputs, _ = model(signal, masktoken=masktoken)
        loss1, loss2 = reconstructionloss(outputs, label, masktoken)
        loss3 = frequencyloss(outputs, label)
        loss = weight[0] * loss1 + weight[1] * loss2 + weight[2] * loss3
        
        totalloss[0] += loss1.item() * weight[0]
        totalloss[1] += loss2.item() * weight[1]
        totalloss[2] += loss3.item() * weight[2]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    print(f'global reconstruction loss {totalloss[0]/(len(dataloader) * dataloader.batch_size) :.5f}')
    print(f'masked reconstruction loss {totalloss[1]/(len(dataloader) * dataloader.batch_size) :.5f}')
    print(f'frequency loss {totalloss[2]/(len(dataloader) * dataloader.batch_size) :.5f}')
    return sum(totalloss)


def reconstructionloss(signal1, signal2, masktoken):
    assert signal1.shape==signal2.shape
    batchsize, signallen = signal1.shape
    # 给每个patch计算损失
    patchnum = masktoken.shape[-1]
    losses = []
    for j in range(patchnum):
        start = j*signallen//patchnum
        end = (j+1)*signallen//patchnum
        losses.append(torch.mean((signal1[:, start:end] - signal2[:, start:end]).abs() ** 2))
    losses = torch.stack(losses).cuda()
    
    # 整个序列的损失 = 每个patch损失之和
    loss1 = torch.sum(losses) 
    # 掩码还原的损失 = 掩码的patch损失之和
    loss2 = torch.sum(torch.mul(losses, masktoken))
    return loss1, loss2


def frequencyloss(signal1, signal2):
    assert signal1.shape==signal2.shape
    signal1 = torch.abs(torch.fft.fft(signal1))
    signal2 = torch.abs(torch.fft.fft(signal2))
    return torch.mean((signal1-signal2)**2)

    
if __name__ == '__main__':
    # dataset
    config = {#'CRWU': '/root/autodl-tmp/CRWU-mat/CRWU',
              'HUST': '/root/autodl-tmp/HUST bearing/HUST bearing',
              'LinGang': '/root/autodl-tmp/LinGang'}
    weight, patchnum, mask_ratio = [1,1,0.5], 64, 0.5
    wtlevel = 2
    dataset = ALLDataset(list(config.keys()), signallen=1024, pathconfig=config, wtlevel=wtlevel)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)    
    visdataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
    # transformer model
    model = Transformer(in_channels=wtlevel+1,
                        d_model=d_model,
                        max_len=max_len,
                        ffn_hidden=ffn_hidden,
                        n_head=n_heads,
                        n_layers=n_layers,
                        drop_prob=drop_prob,
                        signallength=1024, 
                        patchnum=patchnum,
                        device=device,
                        classnum=None,
                        dec_voc_size=1024).to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)

    # optimizer, scheduler, criterion
    optimizer = Adam(params=model.parameters(),
                     lr=init_lr,
                     weight_decay=weight_decay,
                     eps=adam_eps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     verbose=True,
                                                     factor=factor,
                                                     patience=patience)
    # start training
    for epoch in range(200):
        print(f"epoch: {epoch}, now training")
        time.sleep(1)
        trainloss = train(model, dataloader, optimizer=optimizer, weight=weight, patchnum=patchnum, mask_ratio=mask_ratio)
        if epoch > warmup:
            scheduler.step(trainloss/len(dataloader))
        
        if epoch%5==0:
            torch.save(
                {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()},
                 '/root/MAE/snapshot/checkpoint_e%d.pth' % (epoch))
            # visualize
            visualize(visdataloader, model, savefolder=f"/root/MAE/results/pics-epoch{epoch}/", patchnum=patchnum, mask_ratio=mask_ratio, num=30)
        
        with open('/root/MAE/logs.txt', 'a') as file:
            try:
                file.write(f"epoch:{epoch}, learning rate:{optimizer.param_groups[-1]['lr']}, trainloss:{trainloss/len(dataloader)}\n")
            except:
                file.write(f"epoch:{epoch}, trainloss:{trainloss/len(dataloader)}")
        
        
        
        
        

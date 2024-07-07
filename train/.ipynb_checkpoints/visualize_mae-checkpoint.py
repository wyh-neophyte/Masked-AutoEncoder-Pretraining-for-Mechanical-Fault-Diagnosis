import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from conf import *
import torch
from torch import nn
from models.model.transformer import Transformer
from train.masktokens import *
from data.MAEDataset import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize(dataloader, model, savefolder, patchnum, mask_ratio, num):
    model.eval()
    picnum = 0
    for signal, label in tqdm(dataloader):
        masktoken = masktokens(patchnum=patchnum, mask_ratio=mask_ratio)
        signal, label, masktoken = signal.cuda().float(), label.cuda().float(), masktoken.cuda().long()
        outputs, _ = model(signal, masktoken=masktoken)
        
        masked_signal = vismasktokens(signal, signallen=1024, masktoken=masktoken).squeeze(0).cpu().detach()
        outputs, signal, label = outputs.squeeze(0).cpu().detach(), signal.squeeze(0).cpu().detach(), label.squeeze(0).cpu().detach()
        C = masked_signal.shape[0]

        plt.figure(figsize=(20, 28))
        for i in range(0, C):
            #print(f"painting Masked Signal Components {i+1}")
            # masked signal
            plt.subplot(C+2,2, 2*i+1)
            plt.plot(list(range(0,1024)), masked_signal[i].numpy(), label='Signal', color='blue')
            plt.title(f'Masked Signal Components {i+1}')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.ylim(-1, 1)
            
        for i in range(0, C):
            #print(f"painting UnMasked Signal Components {i+1}")
            # unmasked signal
            plt.subplot(C+2,2, 2*i+2)
            plt.plot(list(range(0,1024)), signal[i].numpy(), label='Signal', color='blue')
            plt.title(f'Unmasked Signal Components {i+1}')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.ylim(-1, 1)
            
        # outputs
        plt.subplot(C+2,2,2*C+1)
        plt.plot(list(range(0,1024)), outputs.numpy(), label='Outputs', color='red')
        plt.title('Outputs')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.ylim(-1, 1)
        
        # label
        plt.subplot(C+2,2,2*C+2)
        plt.plot(list(range(0,1024)), label.numpy(), label='Label', color='green')
        plt.title('Ground Truth Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.ylim(-1, 1)
        
        frequency = torch.fft.fft(outputs)
        xaxis = torch.arange(len(frequency))
        plt.subplot(C+2,2,2*C+3)
        plt.plot(xaxis.numpy(), torch.abs(frequency).numpy())
        plt.title('Output Signal Frequncy')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        
        frequency = torch.fft.fft(label)
        xaxis = torch.arange(len(frequency))
        plt.subplot(C+2,2,2*C+4)
        plt.plot(xaxis.numpy(), torch.abs(frequency).numpy())
        plt.title('Ground Truth Signal Frequncy')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        
        os.makedirs(savefolder, exist_ok=True)
        plt.savefig(savefolder+str(picnum)+".jpg")
        plt.close()
        picnum += 1
        if picnum>num:
            break
        
        
        
        

    


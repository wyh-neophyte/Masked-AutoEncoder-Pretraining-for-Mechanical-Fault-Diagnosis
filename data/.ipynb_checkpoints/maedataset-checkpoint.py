import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.LvmDataset import *
from data.MatDataset import *
import pywt

class ALLDataset(Dataset):
    def __init__(self, datasets, signallen, pathconfig, wtlevel=3):
        self.data = []
        self.num = 0
        self.signallen = signallen
        self.wtlevel=wtlevel
        datasetlens = []
        for datasetname in datasets:
            datasetpath = pathconfig[datasetname]
            data = self.readdata(datasetname, datasetpath)
            if datasetname.lower()=='crwu':
                datasetlens.append(sum([item.shape[0] for item in data])*0.5)  # crwu较为简单，因此我设置重要程度为0.5, 减少crwu在训练集中的比例
            else:
                datasetlens.append(sum([item.shape[0] for item in data]))
            self.data.append(data)
            self.num += sum([item.shape[0]//self.signallen for item in data])
        # 根据比例, 选择数据集, 数据少的数据集被选的概率较小, 避免训练数据中有大量重复
        self.datasetid = list(range(len(datasets)))
        self.datasetprob = [len/sum(datasetlens) for len in datasetlens]

    def readdata(self, datasetname, datasetpath):
        if datasetname.lower() == "crwu":
            data = CRWU_loadmat(datasetpath)
        elif datasetname.lower() == "hust":
            data = HUST_loadmat(datasetpath)
        elif datasetname.lower() == "lingang":
            data = LinGang_loadlvm(datasetpath)
        else:
            return NotImplementedError
        return data

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        datasetid = np.random.choice(self.datasetid, p=self.datasetprob)
        seqid = idx % len(self.data[datasetid])
        signal = self.data[datasetid][seqid].squeeze(1)
        # 截取一段信号
        start = random.randint(0, signal.shape[0]-self.signallen)
        signal = signal[start: start+self.signallen]
        # 最大-最小归一化信号
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        label = torch.tensor(signal)
        coeffs = pywt.wavedec(signal, 'db4', level=self.wtlevel)
        reconstructed_signals = []
        for i in range(0, len(coeffs)):
            coeffs_temp = [np.zeros_like(c) for c in coeffs]
            coeffs_temp[i] = coeffs[i]
            reconstructed_signal = pywt.waverec(coeffs_temp, 'db4')
            reconstructed_signals.append(torch.tensor(reconstructed_signal))
        signal = torch.stack(reconstructed_signals, dim=0)
        return signal, label

import lvm_read
import os
import random
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pywt

class LinGangTestDataset(Dataset):
    def __init__(self, datasetpath, load=None, speed=None, signallen=1024, datatype='train', ratio=0.8, wtlevel=2):
        self.labels = {'ball': 0, 'holder': 1, 'inner': 2, 'normal': 3, 'outer': 4}
        self.wtlevel=wtlevel
        self.datadict = {}
        filepaths = glob.iglob(datasetpath + '/**/channel2_002.lvm', recursive=True)
        for filepath in filepaths:
            if load is not None:
                sign = 0
                for loaditem in load:
                    if loaditem in filepath:
                        sign=1
                if sign==0:
                    continue

            if speed is not None:
                sign = 0
                for speeditem in speed:
                    if speeditem in filepath:
                        sign=1
                if sign==0:
                    continue
            faultclass = os.path.dirname(filepath).split("-")[-1]
            lvm_data = lvm_read.read(filepath)
            maxkey = [key for key in lvm_data.keys() if type(key) is int]
            #print(maxkey, type(maxkey[0]))
            maxkey = max(maxkey)
            #print(maxkey)
            for key in lvm_data.keys():
                if type(key) is int:
                    if datatype=='train':
                        if key > maxkey*ratio:
                            continue
                    elif datatype=='val':
                        if key < maxkey*ratio:
                            continue
                    idx = len(self.datadict)
                    signal = lvm_data[key]['data'] / 1.0
                    try:
                        start = random.randint(0, signal.shape[0] - signallen)
                        self.datadict[str(idx)] = (signal[start:start + signallen].squeeze(-1), self.labels[faultclass])
                    except:
                        pass
    def __len__(self):
        return len(self.datadict)

    def __getitem__(self, idx):
        signal, label = self.datadict[str(idx)]
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        coeffs = pywt.wavedec(signal, 'db4', level=self.wtlevel)
        reconstructed_signals = []
        for i in range(0, len(coeffs)):
            coeffs_temp = [np.zeros_like(c) for c in coeffs]
            coeffs_temp[i] = coeffs[i]
            reconstructed_signal = pywt.waverec(coeffs_temp, 'db4')
            reconstructed_signals.append(torch.tensor(reconstructed_signal))
        signal = torch.stack(reconstructed_signals, dim=0)
        return signal, label

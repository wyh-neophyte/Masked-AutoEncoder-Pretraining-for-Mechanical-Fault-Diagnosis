import lvm_read
import os
import glob
import torch

def LinGang_loadlvm(datasetpath):
    data = []
    filepaths = glob.iglob(datasetpath + '/**/channel2_002.lvm', recursive=True)
    for filepath in filepaths:
        lvm_data = lvm_read.read(filepath)
        for key in lvm_data.keys():
            if type(key) is int:
                if lvm_data[key]['data'].shape[0]>1024:
                    data.append(lvm_data[key]['data'])
                else:
                    print("deprecated for length fewer than 1024")
    return data
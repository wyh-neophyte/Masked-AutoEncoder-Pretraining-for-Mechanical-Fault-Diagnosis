import torch
import glob
import scipy


def CRWU_loadmat(datasetpath):
    data = []
    filepaths = glob.iglob(datasetpath + '/**/*.mat', recursive=True)
    for filepath in filepaths:
        mat_data = scipy.io.loadmat(filepath)
        for key in mat_data.keys():
            if 'DE' in key:
                DE = mat_data[key]
                data.append(DE)
            elif 'FE' in key:
                FE = mat_data[key]
                data.append(FE)
            elif 'BA' in key:
                BA = mat_data[key]
                data.append(BA)
    return data


def HUST_loadmat(datasetpath):
    data = []
    filepaths = glob.iglob(datasetpath + '/**/*.mat', recursive=True)
    for filepath in filepaths:
        mat_data = scipy.io.loadmat(filepath)
        for key in mat_data.keys():
            if 'ru' in key:
                ru = mat_data[key]
                data.append(ru)
            elif 'ru_raw' in key:
                ru_raw = mat_data[key]
                data.append(ru_raw)
    return data


def PU_loadmat(datasetpath):
    data = []
    filepaths = glob.iglob(datasetpath + '/**/*.mat', recursive=True)
    for filepath in filepaths:
        mat_data = scipy.io.loadmat(filepath)
        for key in mat_data.keys():
            if '__' in key:
                pass
            else:
                data.append(mat_data[key])
    return data


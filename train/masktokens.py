import torch
import random


def masktokens(patchnum, mask_ratio):
    mask_token = [0]*patchnum  
    masknum = int(patchnum * mask_ratio)
    mask_positions = random.sample(range(patchnum), masknum)
    for pos in mask_positions:
        mask_token[pos] = 1
    mask_token = torch.tensor(mask_token).long()
    return mask_token


def vismasktokens(inputs, signallen, masktoken):
    outputs = inputs.clone()  # 1, channel, L
    channel = outputs.shape[1]
    patchnum = masktoken.shape[-1]
    startids = [startid*(signallen//patchnum) for startid in range(patchnum)]
    for idx in range(patchnum):
        mask = masktoken[idx]
        if mask:
            start = startids[idx]
            end = start + signallen//patchnum
            for i in range(channel):
                outputs[:, i, start: end] = torch.mean(outputs[:, i, start: end])
    return outputs
import torch

def patchify(x_tensor, patchnum):
    """
    params x_tensor: tensor in shape (B, C, N)
    params patchnum: int
    return:
    """
    B, C, N = x_tensor.shape
    assert N % patchnum == 0, "the length must be devided by patchnum to pachify"
    x_tensor = x_tensor.view(B, C, patchnum, N // patchnum)
    return x_tensor

def mask(x_tensor, masktoken):
    """
    params x_tensor: tensor in shape (B, C, patchnum, N//patchnum)
    params masktoken: tensor in shape (patchnum), 0 denotes unmasked, 1 denotes masked
    return: tensor in shape (B, unmasked patchnum, N per patch)
    """
    mask_indices = torch.where(masktoken == 0)[0]
    return x_tensor[:, :, mask_indices, :]


def mask_pos(x_tensor, masktoken):
    """
    params x_tensor: tensor in shape (B, patchnum, N//patchnum)
    params masktoken: tensor in shape (patchnum), 0 denotes unmasked, 1 denotes masked
    return: tensor in shape (B, unmasked patchnum, N per patch)
    """
    mask_indices = torch.where(masktoken == 0)[0]
    return x_tensor[:, mask_indices, :]


def unpatchify(x_tensor, blank_token, patchnum, masktoken, device):
    B, _, C = x_tensor.shape
    full_tensor = torch.zeros(size=(B, patchnum, C)).to(device)
    mask_indices = torch.where(masktoken == 0)[0]
    full_tensor[:, mask_indices, :] = x_tensor
    
    mask_indices2 = torch.where(masktoken == 1)[0]
    full_tensor[:, mask_indices2, :] = torch.mean(x_tensor, dim=1).unsqueeze(1)
    full_tensor[:, mask_indices2, :] += blank_token
    return full_tensor


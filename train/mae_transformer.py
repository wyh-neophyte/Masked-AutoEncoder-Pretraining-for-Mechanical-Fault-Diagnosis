import os
import tqdm
import torch
from torch import nn, optim
from torch.optim import Adam
import torch.nn.functional as F
from conf import *
from data.MAEDataset import *
from models.model.transformer import Transformer
from train.visualize_mae import *
from train.masktokens import *

import argparse
parser = argparse.ArgumentParser(description='MAE pretraining for machenical fault diagnosis')
parser.add_argument('--datasets', type=list, default=['CRWU', 'HUST'],
                    help='the datasets for MAE pretraining')
parser.add_argument('--datapath', type=str, default='',
                    help='path to dataset folder')
parser.add_argument('--savepath', type=str, default='',
                    help='path to save checkpoints')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--patchnum', type=int, default=64)
parser.add_argument('--mask_ratio', type=float, default=0.5)
parser.add_argument('--wtlevel', type=int, default=2,
                    help='wave transform level')
parser.add_argument('--weight', type=list, default=[1, 1, 0.3],
                    help='the weight of three losses, i.e.,'
                         'reconstruction loss on the whole signal,'
                         'reconstruction loss on masked patches,'
                         'the frequency alignment loss')
parser.add_argument('--visualize', type=bool, default=True,
                    help='whether to visualize')
args = parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


def train(model, dataloader, optimizer, weight=[1, 1, 1], patchnum=32, mask_ratio=0.50):
    totalloss = [0, 0, 0]
    for signal, label in tqdm.tqdm(dataloader):
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
    print(f'global reconstruction loss {totalloss[0] / (len(dataloader) * dataloader.batch_size) :.5f}')
    print(f'masked reconstruction loss {totalloss[1] / (len(dataloader) * dataloader.batch_size) :.5f}')
    print(f'frequency loss {totalloss[2] / (len(dataloader) * dataloader.batch_size) :.5f}')
    return sum(totalloss)


def reconstructionloss(signal1, signal2, masktoken):
    assert signal1.shape == signal2.shape
    batchsize, signallen = signal1.shape
    # calculate the reconstruction loss on all patches (MSELoss)
    patchnum = masktoken.shape[-1]
    losses = []
    for j in range(patchnum):
        start = j * signallen // patchnum
        end = (j + 1) * signallen // patchnum
        losses.append(torch.mean((signal1[:, start:end] - signal2[:, start:end]).abs() ** 2))
    losses = torch.stack(losses).cuda()

    # loss1 = the sum of reconstruction loss on all patches
    loss1 = torch.sum(losses)
    # loss2 = the sum of reconstruction loss on masked patches
    loss2 = torch.sum(torch.mul(losses, masktoken))
    return loss1, loss2


def frequencyloss(signal1, signal2):
    assert signal1.shape == signal2.shape
    signal1 = torch.abs(torch.fft.fft(signal1))
    signal2 = torch.abs(torch.fft.fft(signal2))
    return torch.mean((signal1 - signal2) ** 2)


def main():
    batch_size, weight, patchnum, mask_ratio, wtlevel = args.batchsize, args.weight, args.patchnum, args.mask_ratio, args.wtlevel
    # dataset and dataloader
    config = {}
    for dataset in args.datasets:
        config[dataset] = os.path.join(args.datapath, dataset)
    dataset = ALLDataset(list(config.keys()), signallen=1024, pathconfig=config, wtlevel=wtlevel)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    if args.visualize:
        visdataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # transformer model
    model = Transformer(in_channels=wtlevel + 1,
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
    for epoch in range(args.epoch):
        print(f"epoch: {epoch}, now training")
        time.sleep(1)
        trainloss = train(model, dataloader, optimizer=optimizer, weight=weight, patchnum=patchnum,
                          mask_ratio=mask_ratio)
        if epoch > warmup:
            scheduler.step(trainloss / len(dataloader))

        if epoch % 20 == 0:
            torch.save(
                {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()},
                '/root/MAE/snapshot/checkpoint_e%d.pth' % (epoch))
            if args.visualize:
                # visualize
                visualize(visdataloader, model, savefolder=f"/root/MAE/results/pics-epoch{epoch}/", patchnum=patchnum,
                          mask_ratio=mask_ratio, num=30)

        # save training
        with open('/root/MAE/logs.txt', 'a') as file:
            file.write(f"epoch:{epoch}, learning rate:{optimizer.param_groups[-1]['lr']}, \
                         trainloss:{trainloss / len(dataloader)}\n")


if __name__ == '__main__':
    main()
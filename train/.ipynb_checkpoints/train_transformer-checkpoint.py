import sys

from tqdm import tqdm
from data.normaldataset import *
from conf import *
from models.model.transformer import Transformer
from torch import nn, optim
from torch.optim import Adam
import torch.nn.functional as F
import time
import argparse

parser = argparse.ArgumentParser(description='train transformer')
parser.add_argument('--mae', type=bool, default=True)
parser.add_argument('--mae_pretrained', type=str, default='/root/MAE/snapshot/checkpoint_e60.pth')
parser.add_argument('--train_head_only', type=bool, default=True)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--train_speed', type=int, default=0)
parser.add_argument('--test_speed', type=list, default=['1200'])
args = parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


def train(model, dataloader, optimizer, criterion):
    model.train()
    model.requires_grads=True
    if args.train_head_only:
        model.encoder.requires_grads=False
    for signal, label in tqdm(dataloader):
        signal, label = signal.cuda().float(), label.cuda().long()
        outputs = model(signal)
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, criterion):
    # 训练后
    model.eval()
    model.requires_grads=False
    success_num = 0
    total_loss = 0
    for signal, label in tqdm(dataloader):
        signal, label = signal.cuda().float(), label.cuda().long()
        outputs = model(signal)
        loss = criterion(outputs, label)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        success_num += (predicted_class == label).sum()
        total_loss += loss.item()
    print(f"success: {success_num / (len(dataloader) * dataloader.batch_size) :.3f},\
            loss: {total_loss/(len(dataloader) * dataloader.batch_size):.3f}")
    return total_loss/(len(dataloader) * dataloader.batch_size), success_num / (len(dataloader) * dataloader.batch_size)


def main():
    wtlevel = 2
    patchnum = 64
    
    datasetpath = '/root/autodl-tmp/LinGang'
    if args.train_speed == 0:
        train_speed = ['1800']
    elif args.train_speed == 1:
        train_speed = ['1800', '1500']
    # train dataset and dataloader
    traindataset = LinGangTestDataset(datasetpath, speed=train_speed, datatype='train', ratio=0.8, wtlevel=wtlevel)
    trainDataloader = DataLoader(traindataset, batch_size=180, shuffle=True)
    # val dataset and dataloader
    valdataset = LinGangTestDataset(datasetpath, speed=train_speed, datatype='val', ratio=0.8, wtlevel=wtlevel)
    valDataloader = DataLoader(valdataset, batch_size=180, shuffle=True)
    # test dataset and dataloader
    testdataset = LinGangTestDataset(datasetpath, speed=args.test_speed, wtlevel=wtlevel)
    testDataloader = DataLoader(testdataset, batch_size=180, shuffle=True)
    
    # savepath
    if args.mae:
        savepath = 'mae_' + args.mae_pretrained.split('_')[-1][:-4]
        if args.train_head_only:
            savepath += '_headonly'
        savepath += '_train'+''.join(train_speed)+'_test'+''.join(args.test_speed)
    else:
        savepath = 'normal'
        if args.train_head_only:
            savepath += '_headonly'
        savepath += '_train'+''.join(train_speed)+'_test'+''.join(args.test_speed)
    savepath = os.path.join('/root/MAE/train/results', savepath)
    print(args)
    print(savepath)
    

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
                        classnum=5).to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    #model.apply(initialize_weights)
    
    if args.mae:
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.mae_pretrained)['state_dict'], strict=False)
        print("missing keys", missing_keys)
        #print("unexpected keys", unexpected_keys)

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.layers[-2:].parameters():
        param.requires_grad = True
        
    print(model)
    # optimizer, scheduler, criterion
    optimizer = Adam(params=model.parameters(),
                     lr=init_lr,
                     weight_decay=weight_decay,
                     eps=adam_eps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     verbose=True,
                                                     factor=factor,
                                                     patience=patience)
    criterion = nn.CrossEntropyLoss()
    
    trainloss = []
    trainsucc = []
    validloss = []
    validsucc = []
    testloss = []
    testsucc = []
    for epoch in range(args.epoch):
        # train
        print(f"epoch: {epoch}, now training")
        time.sleep(1)
        train(model, trainDataloader, optimizer=optimizer, criterion=criterion)
        
        # train
        print(f"epoch: {epoch}, now evaluating on train dataset")
        time.sleep(1)
        loss, success_rate = evaluate(model, trainDataloader, criterion=criterion)
        trainloss.append(loss)
        trainsucc.append(success_rate.item())
        
        # # validate
        # print(f"epoch: {epoch}, now evaluating on validate dataset")
        # time.sleep(1)
        # loss, success_rate = evaluate(model, valDataloader, criterion=criterion)
        # validloss.append(loss)
        # validsucc.append(success_rate.item())
        
        # scheduler
        if epoch > warmup:
            scheduler.step(loss)
            
        # test
        print(f"epoch: {epoch}, now testing")
        time.sleep(1)
        loss, success_rate = evaluate(model, testDataloader, criterion=criterion)
        testloss.append(loss)
        testsucc.append(success_rate.item())

    
    # save results
    os.makedirs(savepath, exist_ok=True)
    with open(os.path.join(savepath, 'trainloss.txt'), 'a') as file:
        file.writelines([str(item)+"\n" for item in trainloss])
        file.writelines(["\n"])
    with open(os.path.join(savepath, 'trainsucc.txt'), 'a') as file:
        file.writelines([str(item)+"\n" for item in trainsucc])
        file.writelines(["\n"])
    # with open(os.path.join(savepath, 'validloss.txt'), 'a') as file:
    #     file.writelines([str(item)+"\n" for item in validloss])
    #     file.writelines(["\n"])
    # with open(os.path.join(savepath, 'validsucc.txt'), 'a') as file:
    #     file.writelines([str(item)+"\n" for item in validsucc])
    #     file.writelines(["\n"])
    with open(os.path.join(savepath, 'testloss.txt'), 'a') as file:
        file.writelines([str(item)+"\n" for item in testloss])
        file.writelines(["\n"])
    with open(os.path.join(savepath, 'testsucc.txt'), 'a') as file:
        file.writelines([str(item)+"\n" for item in testsucc])
        file.writelines(["\n"])

    torch.save(
            {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()},
             '/root/MAE/train/classifier/checkpoint_final.pth')
    
    
if __name__ == '__main__':
    for _ in range(10):
        args.epoch = 100
        init_lr = 5e-4
        args.mae = False
        args.train_head_only = False
        args.train_speed = 0
        main()

        args.epoch = 100
        init_lr = 5e-4
        args.mae = False
        args.train_head_only = False
        args.train_speed = 1
        main()
        
        args.epoch = 100
        init_lr = 5e-4
        args.mae = True
        args.train_head_only = False
        args.train_speed = 0
        main()
        
        args.epoch = 100
        init_lr = 5e-4
        args.mae = True
        args.train_head_only = False
        args.train_speed = 1
        main()

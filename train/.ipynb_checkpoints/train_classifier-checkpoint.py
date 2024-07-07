import sys
from tqdm import tqdm
from data.ClassifierDataset import *
from conf import *
from models.model.transformer import Transformer
from torch import nn, optim
from torch.optim import Adam
import torch.nn.functional as F
import time
import argparse


parser = argparse.ArgumentParser(description='train transformer')
parser.add_argument('--dataset', type=str, default='LinGang')
parser.add_argument('--mae', type=bool, default=False)
parser.add_argument('--mae_pretrained', type=str, default='./results/MAE/checkpoint_e50.pth')
parser.add_argument('--datapath', type=str, default='',
                    help='path to dataset folder')
parser.add_argument('--savepath', type=str, default='./results',
                    help='path to save checkpoints')
parser.add_argument('--classnum', type=int, default=5,
                    help='LinGang dataset has 5 classes')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--patchnum', type=int, default=64)
parser.add_argument('--wtlevel', type=int, default=2,
                    help='wave transform level')
parser.add_argument('--train_speed', type=list, default=['1800']) #['1800','1500']
parser.add_argument('--test_speed' , type=list, default=['1200'])
args = parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


def train(model, dataloader, optimizer, criterion):
    model.train()
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.layers[-2:].parameters():
        param.requires_grad = True
    for signal, label in tqdm(dataloader):
        signal, label = signal.cuda().float(), label.cuda().long()
        outputs = model(signal)
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, criterion):
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
    epoch, batchsize, patchnum, wtlevel = args.epoch, args.batchsize, args.patchnum, args.wtlevel
    datasetpath = os.path.join(args.datapath, args.dataset)
    # train dataset and dataloader
    traindataset = LinGangTestDataset(datasetpath, speed=args.train_speed, datatype='train', ratio=0.9, wtlevel=wtlevel)
    trainDataloader = DataLoader(traindataset, batch_size=batchsize, shuffle=True)
    # test dataset and dataloader
    testdataset = LinGangTestDataset(datasetpath, speed=args.test_speed, wtlevel=wtlevel)
    testDataloader = DataLoader(testdataset, batch_size=batchsize, shuffle=True)
    
    # savepath
    savepath = args.savepath

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
                        classnum=args.classnum).to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    if args.mae:
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.mae_pretrained)['state_dict'], strict=False)
        print("missing keys", missing_keys)
        
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
    testloss = []
    testsucc = []
    for epoch in range(args.epoch):
        # train
        print(f"epoch: {epoch}, now training")
        time.sleep(1)
        train(model, trainDataloader, optimizer=optimizer, criterion=criterion)
        
        print(f"epoch: {epoch}, now evaluating on train dataset")
        time.sleep(1)
        loss, success_rate = evaluate(model, trainDataloader, criterion=criterion)
        trainloss.append(loss)
        trainsucc.append(success_rate.item())
        
        # scheduler
        scheduler.step(loss)
            
        # test
        print(f"epoch: {epoch}, now testing")
        time.sleep(1)
        loss, success_rate = evaluate(model, testDataloader, criterion=criterion)
        testloss.append(loss)
        testsucc.append(success_rate.item())

    
    os.makedirs(os.path.join(savepath, 'classifier'), exist_ok=True)
    # save results
    os.makedirs(savepath, exist_ok=True)
    with open(os.path.join(savepath, 'classifier/trainloss.txt'), 'a') as file:
        file.writelines([str(item)+"\n" for item in trainloss])
        file.writelines(["\n"])
    with open(os.path.join(savepath, 'classifier/trainsucc.txt'), 'a') as file:
        file.writelines([str(item)+"\n" for item in trainsucc])
        file.writelines(["\n"])
    with open(os.path.join(savepath, 'classifier/testloss.txt'), 'a') as file:
        file.writelines([str(item)+"\n" for item in testloss])
        file.writelines(["\n"])
    with open(os.path.join(savepath, 'classifier/testsucc.txt'), 'a') as file:
        file.writelines([str(item)+"\n" for item in testsucc])
        file.writelines(["\n"])

    torch.save({'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                 os.path.join(savepath, 'classifier/classifier.pth'))
    
    
if __name__ == '__main__':
    main()
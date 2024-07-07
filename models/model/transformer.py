import torch
from torch import nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, max_len, in_channels, ffn_hidden, n_layers, drop_prob, device, 
                 signallength=1024, patchnum=16, classnum=None, dec_voc_size=None):
        super().__init__()
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device,
                               signallength=signallength,
                               patchnum=patchnum,
                               in_channels=in_channels)

        self.classnum = classnum
        # 如果是分类任务
        if classnum is not None and dec_voc_size is None:
            self.classifier = nn.Sequential(nn.Linear(d_model * patchnum, 1000),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(1000, classnum)
                                            )
            if 'cuda' in device:
                self.classifier = self.classifier.cuda()
        # 如果是masked autoencoder预训练
        elif classnum is None and dec_voc_size is not None:
            self.decoder = Decoder(d_model=d_model,
                                   n_head=n_head,
                                   max_len=max_len,
                                   ffn_hidden=ffn_hidden,
                                   drop_prob=drop_prob,
                                   n_layers=n_layers,
                                   device=device,
                                   patchnum=patchnum,
                                   dec_voc_size=dec_voc_size)
        else:
            raise NotImplementedError

    def forward(self, src, masktoken=None, returnfeature=False):
        enc_src = self.encoder(src, masktoken=masktoken, src_mask=None)  # B, unmasked patch num, C
        if self.classnum is not None:
            B = enc_src.shape[0]
            features = enc_src.reshape(B, -1)
            if returnfeature:
                return self.classifier(features), features
            else:
                return self.classifier(features)
        else:
            assert masktoken is not None
            output = self.decoder(enc_src, masktoken=masktoken, trg_mask=None, src_mask=None)
            return output, self.encoder.masked_x_patch


if __name__ == '__main__':
    #signal = torch.randn(size=(8, 4, 1024)).cuda()
    signal = torch.randn(size=(8,4, 1024)).cuda()
    EncoderDecoder = Transformer(d_model=192, n_head=4, max_len=64, in_channels=4, ffn_hidden=128, n_layers=1, drop_prob=0.1,
                                 signallength=1024, patchnum=8, device='cuda', classnum=None, dec_voc_size=1024)
    masktoken = torch.tensor([0,0,0,0,0,1,0,1]).cuda()
    output = EncoderDecoder(signal, masktoken=masktoken)
    print(output[0].shape, output[1].shape)

    Classifier = Transformer(d_model=512, n_head=4, max_len=64, in_channels=4, ffn_hidden=128, n_layers=1, drop_prob=0.1,
                             device='cuda', classnum=5)
    output = Classifier(signal)
    print(output.shape)

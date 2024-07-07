"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.positional_encoding import PositionalEncoding
from models.utils.utils import unpatchify

class Decoder(nn.Module):
    def __init__(self, max_len, patchnum, d_model, ffn_hidden, n_head, n_layers, drop_prob, device, dec_voc_size=1024):
        super().__init__()
        self.emb = PositionalEncoding(d_model=d_model,
                                      max_len=max_len,
                                      device=device)
        self.patchnum = patchnum
        self.device = device
        self.blanktoken = nn.Parameter(torch.zeros(size=(d_model,))).to(device)
        
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob).to(device)
                                     for _ in range(n_layers)])
        self.patchsize = dec_voc_size//patchnum
        

        self.projection = nn.Sequential(nn.Linear(d_model, self.patchsize)).to(device)


    def forward(self, enc_src, masktoken, trg_mask=None, src_mask=None):
        # unpatchify
        # (B, unmasked patchnum, d_model) --> (B, patchnum, d_model)
        trg = unpatchify(enc_src, self.blanktoken, self.patchnum, masktoken, device=self.device)
        
        # add position embedding
        pos_embed = self.emb(self.patchnum).unsqueeze(0)
        trg += pos_embed
        
        # transformer decoder
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        # trg (B,patchnum,C) --> trg (B, patchnum, N_patch) --> trg (B, N)
        B, patchnum, C = trg.shape
        trg = trg.reshape(-1, C)
        output = self.projection(trg).reshape(B, patchnum, -1).flatten(start_dim=1)

        return output
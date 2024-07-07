from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.positional_encoding import PositionalEncoding
from models.utils.utils import patchify, mask, mask_pos

class Encoder(nn.Module):

    def __init__(self, max_len, signallength, patchnum, in_channels, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = PositionalEncoding(d_model=d_model,
                                      max_len=max_len, 
                                      device=device)
        
        self.patchsize = signallength//patchnum
        self.patchnum = patchnum
        self.projection = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=self.patchsize, stride=1, padding=0),
                                        nn.BatchNorm1d(d_model),
                                        nn.LeakyReLU(inplace=True),
                                        ).to(device)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob).to(device)
                                     for _ in range(n_layers)])

    def forward(self, x, masktoken, src_mask=None):
        # pachify 
        # (B, C, N) --> (B, C, patchnum, N//patchnum)
        x_patch = patchify(x, self.patchnum)
        
        # add mask 
        # --> (B, C, unmasked patchnum, N//patchnum)
        if masktoken is not None:
            masked_x_patch = mask(x_patch, masktoken)
        else:
            masked_x_patch = x_patch
            
        # reshape tensor 
        # (B, C, unmasked patchnum, N//patchnum) --> (B, unmasked patchnum, C, N//patchnum) --> (B*unmasked patchnum, C, N//patchnum)
        B, C, unmasked_patchnum, N_patch = masked_x_patch.shape 
        masked_x_patch = masked_x_patch.permute(0,2,1,3).reshape(-1, C, N_patch)
        self.masked_x_patch = masked_x_patch.clone()  # need to return in transformer.py
        
        # extract features
        # (B*unmasked patchnum, C, N//patchnum) --> (B*unmasked patchnum, d_model, 1) --> (B, unmasked patchnum, d_model)
        features = self.projection(masked_x_patch)
        features = features.view(B, unmasked_patchnum, -1)
        
        # add position encoding
        pos_embed = self.emb(self.patchnum).unsqueeze(0)
        if masktoken is not None:
            masked_pos_embed = mask_pos(pos_embed, masktoken)
        else:
            masked_pos_embed = pos_embed
        features = features + masked_pos_embed
        
        # transformer encoder layers
        for layer in self.layers:
            features = layer(features, src_mask)

        return features
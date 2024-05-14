import torch
import torch.nn as nn


class SliceNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_type = ['infer', 'loss']
        self.cfg = cfg
        self.encoder_in_dim = 3

        self.encoder = resnet18d(pretrained=False, in_chans=self.encoder_in_dim)


        #----------------------------------------------------
        self.norm = nn.LayerNorm(512)
        self.pos_embed = nn.Parameter(
            positional_encoding(32, 512)
        )
        self.decoder = nn.Sequential(
            TransformerBlock(512,8),
            TransformerBlock(512,8),
            TransformerBlock(512,8),
        )

        #----------------------------------------------------
        self.slice_logit = nn.Linear(512, 1) 


    def infer(self, image):
        batch_size, D, H, W = image.shape
        x = image.reshape(batch_size*D//self.encoder_in_dim, self.encoder_in_dim, H, W )

        f = self.encoder.forward_features(x)
        _,d,h,w = f.shape
        pool = F.adaptive_avg_pool2d(f, 1)
        pool = pool.flatten(1)
        p = pool.reshape(batch_size,-1,d)

        #------
        embed = self.norm(p) + self.pos_embed
        d = self.decoder(embed)

        slice_logit = self.slice_logit(d).squeeze(-1)
        slice_prob = F.sigmoid(slice_logit)
        slice_prob = F.interpolate(slice_prob.unsqueeze(1), size=(D,), mode='linear', align_corners=False).squeeze(1)

        return slice_prob


class MainNet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.liver_logit = nn.Linear()
        self.spleen_logit = nn.Linear()
        self.kidney_logit = nn.Linear()

    def forward(self, batch):

        image = batch['image']
        batch_size, D, H, W = image.shape  #(B, 96, 256, 256)

        ... 
        f = self.encoder.forward_features(x)
        ....
        ....
        f = self.decoder(f)
        flatten = f.mean(dim=[2,3,4]) # pool
        split = torch.split_with_sizes(flatten, batch['num_series'])
        pool  = torch.stack([
            p.max(0)[0] + p.mean(0)
            for p in split])

        liver_logit  = self.liver_logit(pool)
        spleen_logit = self.spleen_logit(pool)
        kidney_logit = self.kidney_logit(pool)
        ...
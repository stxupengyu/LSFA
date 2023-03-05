
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import uniform, normal

class FeatsVAE(nn.Module):
    def __init__(self, args, hidden_dim=512):
        super(FeatsVAE, self).__init__()
        self.input_dim = args.feat_size
        self.latent_dim = args.feat_size
        self.linear = nn.Sequential(
            nn.Linear(self.input_dim +self.input_dim, hidden_dim),
            #nn.LeakyReLU(),
            #nn.Linear(4096, 4096),
            nn.LeakyReLU())
        self.linear_mu =  nn.Sequential(
            nn.Linear(hidden_dim, self.latent_dim),
            nn.ReLU())
        self.linear_logvar =  nn.Sequential(
            nn.Linear(hidden_dim, self.latent_dim),
            nn.ReLU())
        self.generator = nn.Sequential(
            nn.Linear(2*self.latent_dim, hidden_dim),
            nn.LeakyReLU(),
            #nn.Linear(4096, 4096),
            #nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.input_dim),
            #nn.Sigmoid(),
        )
        self.bn1 = nn.BatchNorm1d(self.input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.z_dist = normal.Normal(0, 1)
        self.init_weights()


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # remove abnormal points
        return mu + eps*std

    def init_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Linear):
              m.weight.data.normal_(0, 0.02)
              m.bias.data.normal_(0, 0.02)

    def forward(self, x, p):
        x = torch.cat((x, p), dim=1)
        x = self.linear(x)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)

        latent_feats = self.reparameterize(mu, logvar)

        #Z = self.z_dist.sample(attr.shape).cuda()
        concat_feats = torch.cat((latent_feats, p), dim=1)
        recon_feats = self.generator(concat_feats)
        recon_feats = self.relu(self.bn1(recon_feats))
        return mu, logvar, recon_feats



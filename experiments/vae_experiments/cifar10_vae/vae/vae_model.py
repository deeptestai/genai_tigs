import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, c_num=3, h_dim=4000, z_dim=1024):
        super(ConvVAE, self).__init__()
        self.z_dim = z_dim
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(c_num, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.enc_linear1 = nn.Linear(512 * 4 * 4, h_dim)
        self.enc_linear2 = nn.Linear(h_dim, 2 * z_dim)

        # Decoder
        self.dec_linear1 = nn.Linear(z_dim, h_dim)
        self.dec_linear2 = nn.Linear(h_dim, 512 * 4 * 4)
        self.dec_conv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(64, c_num, 3, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(-1, 512 * 4 * 4)
        h = F.relu(self.enc_linear1(x))
        pre_z = self.enc_linear2(h)
        mu = pre_z[:, :self.z_dim]
        log_var = pre_z[:, self.z_dim:]
        return mu, log_var

    def decode(self, z):
        h = F.relu(self.dec_linear1(z))
        h = F.relu(self.dec_linear2(h))
        h = h.view(-1, 512, 4, 4)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        x = torch.sigmoid(self.dec_conv4(h))
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        rec_x = self.decode(z)
        return rec_x, mu, log_var

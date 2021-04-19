
import torch.nn.functional as F
from Colour_utils.base_color import *
import math
import numpy as np

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1, padding=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super().__init__()
        self.down_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, upsampling_size=None):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(size=upsampling_size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, padding=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=3, stride=1, padding=1)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConvLayer(BaseColor):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   Down(32,64, stride=1, padding=1))
        self.conv2 = Down(64,128, padding=0)
        self.conv3 = Down(128,256, padding=0)
        self.conv4 = Down(256,512, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        conv1 = self.conv1(self.normalize_l(x))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv1, conv2, conv3, conv4


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=16):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=5, stride=3, padding=2)
                          for _ in range(num_capsules)]) #16 num capsules
    
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)  # => batch size, num_capsules, num feat map per capsule, H feature map, W feature map
        # u = u.view(x.shape[0], num_routes, -1)
        u = u.permute(0,2,3,4,1)
        u = u.view(x.shape[0], u.shape[1]*u.shape[2]*u.shape[3], -1)
        return self.squash(u)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, logits_num=32, num_routes=(32,15,15), num_capsules=16):
        super(DigitCaps, self).__init__()
        
        self.num_routes = num_routes
        self.num_copies = 1
        self.W = nn.Parameter(torch.randn(1, np.prod(self.num_routes), self.num_copies, logits_num, num_capsules))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_copies, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        num_routes = np.prod(self.num_routes)
        b_ij = torch.zeros(1, num_routes, self.num_copies, 1)
        b_ij = b_ij.to(x.device)

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij,dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)  
            else:
                u_j = (c_ij*u_hat)
        return v_j.squeeze(1), u_j
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
    
class Reconstruction(BaseColor):
    def __init__(self, logits_num=32, num_capsules=16, num_routes=(32,15,15), AM = False):
        super(Reconstruction, self).__init__()

        self.AM = AM

        self.color_channels = 2
        self.num_routes = num_routes

        self.W = nn.Parameter(torch.randn(1, np.prod(num_routes), 1, num_capsules, logits_num))

        self.reconstruction_capsules = nn.ModuleList([nn.ConvTranspose2d(in_channels=num_routes[0], out_channels=int(512/num_capsules), kernel_size=5, stride=3, padding=2) for _ in range(num_capsules)])
        bilinear = True
        self.reconstruction_layers_up1 = Up(512+512, 512, bilinear=bilinear, upsampling_size=(44,44))
        self.reconstruction_layers_up2 = Up(256+512, 256, bilinear=bilinear, upsampling_size=(48,48))
        self.reconstruction_layers_up3 = Up(128+256, 128, bilinear=bilinear, upsampling_size=(52,52))
        self.reconstruction_layers_up4 = Up(64+128, 64, bilinear=bilinear, upsampling_size=(56,56))
        self.q = nn.Conv2d(64, 313, kernel_size=1, stride=1, padding=0, bias=False)
        # self.ab = nn.Conv2d(64, 2, kernel_size=1,stride=1,padding=0,bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=False)
        
    def forward(self,x,conv1,conv2,conv3,conv4):
        batch_size = x.size(0)

        W = torch.cat([self.W] * batch_size, dim=0)
        uhat=torch.matmul(W.squeeze(2), x.unsqueeze(2).permute(0,3,1,2))
        uhat = uhat.permute(0,2,1,3)
        uhat = uhat.view(uhat.size(0),uhat.size(1),*self.num_routes)

        # Recombine capsules into a feature map matrix
        # A reconstrution capsule sees as input the output of a previous capsule...
        u_rec = [capsule(uhat[:,ii,:,:,:]) for ii, capsule in enumerate(self.reconstruction_capsules)]
        u_rec = torch.cat(u_rec, dim=1)

        a = 0

        # Go up..
        x = self.reconstruction_layers_up1(u_rec, conv4)
        x = self.reconstruction_layers_up2(x, conv3)
        x = self.reconstruction_layers_up3(x, conv2)
        x = self.reconstruction_layers_up4(x, conv1)
        q = self.q(x)
        x = self.model_out(self.softmax(q))

        return  self.unnormalize_ab(self.upsample4(x)),self.softmax(q) #da provare prima era solo q

class CapsNet_MR(nn.Module):
    def __init__(self,logits_num, AM = False, num_capsules=16, num_routes=(32,15,15)):
        super(CapsNet_MR, self).__init__()

        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps(num_capsules=num_capsules)
        self.digit_capsules = DigitCaps(logits_num=logits_num, num_routes=num_routes, num_capsules=num_capsules)
        self.reconstruction = Reconstruction(logits_num=logits_num,AM=AM,num_routes=num_routes,num_capsules=num_capsules)

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        conv1,conv2,conv3,conv4 = self.conv_layer(data)
        primary_caps_output = self.primary_capsules(conv4)
        output,u_hat = self.digit_capsules(primary_caps_output)
        u_hat = u_hat.permute(0,2,3,1,4)
        reconstructionsAB, reconstructionsQ = self.reconstruction(u_hat.squeeze(),conv1,conv2,conv3,conv4)
        return reconstructionsAB, reconstructionsQ
    
    def CE_loss(self,data,preds):
        batch_size = data.size(0)
        loss = -torch.mean(torch.sum(data* torch.log(preds), dim=1))
        return loss  
    
    def loss(self, data, x, target, reconstructions): #<--------------------------------------ML+REC
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)
    
    def loss_togheter(self,data,reconstructions):
        loss_AB = self.mse_loss(reconstructions.view(reconstructions.size(0),-1),data.view(reconstructions.size(0),-1))
        return loss_AB * 0.001
    
    def reconstruction_loss(self,  data, reconstructions,plus=False):
        reconstructions_A = reconstructions[:,0,:,:]
        data_A = data[:,0,:,:]
        reconstructions_B = reconstructions[:,1,:,:]
        data_B = data[:,1,:,:]
        loss_A = self.mse_loss(reconstructions_A.view(reconstructions.size(0), -1), data_A.view(reconstructions.size(0), -1))
        loss_B = self.mse_loss(reconstructions_B.view(reconstructions.size(0), -1), data_B.view(reconstructions.size(0), -1))

        if not plus:
            loss = loss_A + loss_B
        else:
            loss_AB = self.loss_togheter(data,reconstructions)
            loss = loss_AB + loss_A + loss_B

        return loss * 0.001




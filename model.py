# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from util.custom_dataset import FaceLandmarksDataset, Rescale, ToTensor
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class InnerSum(nn.Module):
    def __init__(self):
        super(InnerSum, self).__init__()
    def forward(self, x):
        y = torch.zeros_like(x)
        for i in range(x.size(0)):
            y[i] = x[i].mul(x[i])
        if len(y.shape) == 3:
            return y.sum(2)
        else:
            return y.sum(1)

class ACNN(nn.Module):
    def __init__(self):
        super(ACNN, self).__init__()
        self.inner = InnerSum()
        self.VGG16 = models.vgg16(pretrained=True).features[:21]
        self.PG_base = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),)
        self.PG_attention = nn.Sequential(nn.MaxPool2d(2, stride=2), nn.Conv2d(512, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), View((-1, 128)),
                                          nn.Linear(128, 64), nn.ReLU(),
                                          nn.Linear(64, 1), nn.Sigmoid())
        self.GG_base = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2), nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512), nn.ReLU())
        self.GG_attention = nn.Sequential(nn.MaxPool2d(2, stride=2), nn.Conv2d(512, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), View((-1, 128)),
                                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.PG24_base = nn.ModuleList([self.PG_base for _ in range(24)])
        self.PG24_alpha = nn.ModuleList([self.PG_attention for _ in range(24)])
        self.PG_fc = nn.Linear(512, 64)
        self.PG_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 7)
        self.softmax = nn.Softmax(dim=1)

    def crop_layer(self, img:'(B, C, H, W)', landmarks:'(B, 24, 2)')->list:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pad = nn.ReflectionPad2d(6)  # padding for cropping
        img = pad(img)  # (B, 512, 36, 36)
        total_crop = []
        for i in range(len(landmarks)):
            crop_per_batch = []
            for patch in range(landmarks.size(1)):
                crop_img = img[i][:, (int(landmarks[i, patch, 0]) - 3): (int(
                    landmarks[i, patch, 0]) + 3),
                           (int(landmarks[i, patch, 1]) - 3): (int(
                                        landmarks[i, patch, 1]) + 3)]
                crop_img = crop_img.view(1, 1, 512, 6, 6)
                crop_per_batch.append(crop_img)
            # crop_per_batch = [t.detach().cpu().numpy() for t in crop_per_batch]
            # crop_per_batch = torch.Tensor(crop_per_batch)
            crop_per_batch = torch.cat(crop_per_batch, dim=1)
            total_crop.append(crop_per_batch)
        # total_crop = [t.detach().cpu().numpy() for t in total_crop]
        # total_crop = torch.Tensor(total_crop)
        total_crop = torch.cat(total_crop, dim=0)
        total_crop = total_crop.permute(1, 0, 2, 3, 4)
        return total_crop

    def _branch24(self, crop_img):
        PG_out = []
        for x, base, alpha in zip(crop_img, self.PG24_base, self.PG24_alpha):
            PG_conv2 = base(x)
            PG_reshape = self.PG_avgpool(PG_conv2).view(-1, 512)
            PG_reshape = self.PG_fc(PG_reshape)
            PG_per = PG_reshape * alpha(PG_conv2).view(x.size(0), 1)
            PG_out.append(PG_per)
        return PG_out


    def forward(self, img, landmarks):
        img_feature = self.VGG16(img)  # (B, 512, 28, 28)

        GG_conv2 = self.GG_base(img_feature)
        GG_reshape = nn.AdaptiveAvgPool2d((1, 1))(GG_conv2).view(-1, 512)
        GG_out = GG_reshape * self.GG_attention(GG_conv2).view(img_feature.size(0), 1)

        crop_img = self.crop_layer(img_feature, landmarks)
        PG_out = self._branch24(crop_img)
        PG_total = torch.cat(PG_out, dim=1)
        total_out = torch.cat([GG_out, PG_total], dim=1)

        out = F.relu(self.fc1(total_out))
        out = F.relu(self.fc2(out))
        out = self.softmax(out)
        return out








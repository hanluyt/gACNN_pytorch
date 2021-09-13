# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from util.custom_dataset import FaceLandmarksDataset, Rescale, ToTensor
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torchvision

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
        self.pretrain = models.vgg16(pretrained=True).features[:28]
        self.VGG16 = self.pretrain[:21]
        self.PG_base = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),)
        self.PG_attention = nn.Sequential(nn.MaxPool2d(2, stride=2), nn.Conv2d(512, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), View((-1, 128)),
                                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.GG_base = self.pretrain[21:]
        self.GG_attention = nn.Sequential(nn.MaxPool2d(2, stride=2), nn.Conv2d(512, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), View((-1, 128)),
                                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.PG24_base = nn.ModuleList([self.PG_base for _ in range(24)])
        self.PG24_alpha = nn.ModuleList([self.PG_attention for _ in range(24)])

        self.pad = nn.ReflectionPad2d(6)
        # self.crop = batch_slice(40, 40, 6, 6)
        self.crop = torchvision.ops.roi_pool
        self.PG_fc = nn.Linear(512*6*6, 64)
        self.GG_fc = nn.Linear(512*14*14, 512)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 7)

    # def crop_layer(self, img: '(B, C, H, W)', landmarks: '(B, 24, 2)'):
    #     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     pad = nn.ReflectionPad2d(6)  # padding for cropping
    #     img = pad(img)  # (B, 512, 36, 36)
    #     total_crop = torch.zeros((img.size(0), landmarks.size(1), 512, 6, 6), device=self.device)
    #
    #     for i in range(landmarks.size(0)):  # Batch
    #         # crop_per_batch = []
    #         for patch in range(landmarks.size(1)):  # 24 landmarks
    #             total_crop[i, patch, :, :, :] = img[i, :, (int(landmarks[i, patch, 0]) - 3): (int(
    #                 landmarks[i, patch, 0]) + 3),
    #                                             (int(landmarks[i, patch, 1]) - 3): (int(
    #                                                 landmarks[i, patch, 1]) + 3)]  # crop_img: (512, 6, 6)
    #
    #     total_crop = total_crop.permute(1, 0, 2, 3, 4)  # output: (24, B, 512, 6, 6)
    #     return total_crop

    def _branch24(self, crop_img):
        PG_out = []
        for x, base, alpha in zip(crop_img, self.PG24_base, self.PG24_alpha):
            PG_conv2 = base(x)
            PG_reshape = PG_conv2.view(-1, 512*6*6)
            PG_reshape = self.PG_fc(PG_reshape)
            PG_per = PG_reshape * alpha(PG_conv2).view(x.size(0), 1)
            PG_out.append(PG_per)
        return PG_out

    def forward(self, img, landmarks):
        img_feature = self.VGG16(img)  # (B, 512, 28, 28)
        img_pad = self.pad(img_feature)
        # landmarks = landmarks.long()
        crop_img = self.crop(img_pad, landmarks, output_size=(6, 6))
        crop_img = crop_img.view(24, -1, 512, 6, 6)

        GG_conv2 = self.GG_base(img_feature)
        GG_reshape = GG_conv2.view(-1, 512*14*14)
        GG_reshape = self.GG_fc(GG_reshape)
        GG_out = GG_reshape * self.GG_attention(GG_conv2).view(img_feature.size(0), 1)

        # crop_img = self.crop_layer(img_feature, landmarks)
        PG_out = self._branch24(crop_img)
        PG_total = torch.cat(PG_out, dim=1)
        total_out = torch.cat([GG_out, PG_total], dim=1)

        out = self.fc1(total_out)
        out = F.relu(self.dropout(out))
        out = self.fc2(out)
        return out

def landmark_resize(landmarks:'(B, 24, 2)')->'(B*24, 4)':
    bs = landmarks.size(0)
    batch = list(range(bs))
    batch = np.array(batch * 24).reshape(24, -1).T

    point = np.array(list(range(24)) * bs).reshape(bs, -1)
    insert_point = np.insert(landmarks, 0, point, 2)
    insert_batch = np.insert(insert_point, 0, batch, 2)
    new_landmark = insert_batch.reshape(-1, 4)
    return new_landmark

def data_normal(origin_data, size):  # (-1, 1)
    size = size / 2
    norm_data = origin_data.true_divide(size) - 1
    return norm_data


def grid_field(landmarks, cropsize=6):  # landmarks: (B, 24, 2)
    total_crop = []
    landmarks = landmark_resize(landmarks)  # (B*24, 4)
    lm_batch = landmarks[:, 0].long()
    landmarks_x_l = landmarks[:, 2] - (cropsize / 2)
    landmarks_x_r = landmarks[:, 2] + (cropsize / 2)
    landmarks_y_l = landmarks[:, 3] - (cropsize / 2)
    landmarks_y_r = landmarks[:, 3] + (cropsize / 2)

    for i in range(landmarks.size(0)):
        new_h = torch.linspace(landmarks_x_l[i], landmarks_x_r[i] - 1, cropsize).view(-1, 1).repeat(1, cropsize)
        new_w = torch.linspace(landmarks_y_l[i], landmarks_y_r[i] - 1, cropsize).repeat(cropsize, 1)
        grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)
        grid = data_normal(grid, size=28)
        total_crop.append(grid)
    total_crop = torch.cat(total_crop, dim=0)
    return lm_batch, total_crop

def roi_select(landmarks: '(B, 4, 2)') -> '(B*24, 5)':
    landmarks = landmark_resize(landmarks)
    landmarks_right = landmarks[:, 2:] + 3
    landmarks_left = landmarks[:, 2:] - 3
    landmarks = torch.cat([landmarks[:, 0].view(-1, 1), landmarks_left, landmarks_right], dim=1)

    return landmarks


# if __name__ == '__main__':
#     model = ACNN()
#     shuffle = False
#     device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     train_set = FaceLandmarksDataset(csv_file='train_acnn.csv', root_dir='original/',
#                                      transform=ToTensor())
#     test_set = FaceLandmarksDataset(csv_file='test_acnn.csv', root_dir='original/',
#                                     transform=ToTensor())
#     train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=4, num_workers=0,
#                               pin_memory=True)
#     test_loader = DataLoader(dataset=test_set, shuffle=shuffle, batch_size=4, num_workers=8,
#                              pin_memory=True)
#     for step, batch in enumerate(train_loader):
#         imgs, landmarks, targets = batch['image'], batch['landmarks'] / 8. + 6, batch['label']
#         landmarks = roi_select(landmarks)
#
#         imgs, landmarks, targets = imgs.to(device), landmarks.to(device), targets.to(device)
#         logits = model(imgs, landmarks)
#         print(logits.size())
#         break



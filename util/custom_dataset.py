import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")

def show_landmarks(image, label, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = np.array(self.landmarks_frame.iloc[idx, 1])
        label = label.astype(np.int64)
        landmarks = self.landmarks_frame.iloc[idx, 2:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype(np.float32).reshape(-1, 2)
        sample = {'image': image, 'label': label, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, label, landmarks = sample['image'], sample['label'], sample['landmarks']
        # image = image.astype(np.float32)
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))   # image: 0: 1
        return {'image': img, 'label': label, 'landmarks': landmarks}

class ToTensor(object):
    def __call__(self, sample):
        image, label, landmarks = sample['image'], sample['label'], sample['landmarks']
        # swap axis: numpy image: H x W x C, torch image: C x H x W
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        return {'image': image, 'label': torch.from_numpy(label),
                'landmarks': torch.from_numpy(landmarks)}

def show_landmark_batch(sample_batched):
    images_batch, label_batch, landmarks_batch = sample_batched['image'],\
    sample_batched['label'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size, s=10, marker='.', c='r')
    plt.title('Batch from dataloader')

if __name__ == '__main__':
    face_dataset = FaceLandmarksDataset(csv_file='RAF-DB_test_24.csv', root_dir='../RAF-DB_test/',
                                        transform=transforms.Compose([Rescale((224, 224)), ToTensor()]))
    dataloader = DataLoader(face_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    for i, sample_batched in enumerate(dataloader):
        print(i, sample_batched['image'].size(), sample_batched['label'],
              sample_batched['landmarks'].size())
        if i == 3:
            plt.figure()
            show_landmark_batch(sample_batched)
            plt.axis('off')
            plt.show()
            break











import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class MPI3DHPDataset(Dataset):
    def __init__(self, npz_file_path, root_dir, transform=None):
        self.npz_file = np.load(npz_file_path, allow_pickle=True)
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.npz_file['data'].item()
        self.samples = []

        for subject, sequences in self.data.items():
            for seq_name, seq_data in sequences.items():
                for video_id, frame_data_dict in seq_data[0].items():
                    video_folder = f'video_{video_id}'
                    for frame_index in frame_data_dict.keys():
                        img_path = os.path.join(root_dir, subject, seq_name, 'imageFrames', video_folder, f'{frame_index:06d}.jpg')
                        frame_data = frame_data_dict[frame_index]
                        self.samples.append((img_path, frame_data['data_2d'], frame_data['data_3d']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, data_2d, data_3d = self.samples[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'data_2d': torch.tensor(data_2d), 'data_3d': torch.tensor(data_3d)}

        return sample

# 示例用法
npz_file_path = 'F:/mpi-3dhp/data_test_3dhp.npz'
root_dir = 'F:/mpi-3dhp'
transform = None  # 可以根据需要添加图像变换

dataset = MPI3DHPDataset(npz_file_path, root_dir, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 示例遍历数据
for batch in dataloader:
    images = batch['image']
    data_2d = batch['data_2d']
    data_3d = batch['data_3d']
    print(images.shape, data_2d.shape, data_3d.shape)

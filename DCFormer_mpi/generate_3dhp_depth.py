import torch
import pickle
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import shutil
from depth_anything_v2.dpt import DepthAnythingV2
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from mvn.utils.img import crop_image
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
""" depth anything v2. """
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vitb' # or 'vits', 'vitb', 'vitg'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'depth_anything_v2/checkpoint/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()
total_params = sum(param.numel() for param in model.parameters())
print('Total parameters of depth-anything_v2: {:.2f}M'.format(total_params / 1e6))

transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
def denormalize_screen_coordinates(pt, w, h, is_3d=False):
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    result = pt.copy()
    result[..., :2] = (pt[..., :2] + np.array([1, h / w])) * w / 2
    if is_3d:
        result[..., 2:] = pt[..., 2:] * w / 2
    return np.array(result)

def exec_affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def pre_process(image_path):
    """
    预处理图像的函数
    """
    image = cv2.imread(
        image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    # ##################################
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    # image = image / 255.0
    image = transform ({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    return image

subfolders = ['S1', 'S2', 'S3']
def process_and_save_images(input_folder, output_folder):
     for subfolder in subfolders:
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.exists(subfolder_path):
            for root, dirs, files in os.walk(subfolder_path):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        input_path = os.path.join(root, file)
                        output_path = input_path.replace(input_folder, output_folder)

                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # processed_image = pre_process(input_path) #[1, 3, 686, 518]
                        image = cv2.imread(input_path)

                        with torch.no_grad():
                            depth = model.infer_image(image) #[1, 1, 686, 518]

                        # depth = F.interpolate(depth[None], (ori_h,ori_w), mode='bilinear', align_corners=False)[0, 0] #[256, 192]
                        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                        # depth = depth.cpu().numpy().astype(np.uint8)

                        cv2.imwrite(output_path, depth)

if __name__ == "__main__":
    input_folder ='DCFormer_mpi/dataset/mpi_train'
    output_folder = 'DCFormer_mpi/dataset/mpi_train_depth'
    print("begin")
    process_and_save_images(input_folder, output_folder)

    input_folder_test ='DCFormer_mpi/dataset/mpi_test'
    output_folder_test = 'DCFormer_mpi/datasetmpi_test_depth'
    print("begin")
    process_and_save_images(input_folder_test, output_folder_test)

    print(f"Processed images are saved")
 
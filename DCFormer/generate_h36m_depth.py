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
    image = cv2.imread(
        image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    ori_h,ori_w = image.shape[0],image.shape[1]
    # ##################################
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    # image = image / 255.0
    image = transform ({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    return image, ori_h,ori_w


def process_images_in_directory(labels,input_dir, output_dir):
    output_features = []
    for idx, shot in enumerate(labels):
        # if shot['subject']!=5:
        #     continue
        # load image
        subdir_format = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'
        subdir = subdir_format.format(shot['subject'], shot['action'], shot['subaction'], shot['camera_id']+1)
        image_format = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}_{:06d}.jpg'
        imagename = image_format.format(shot['subject'], shot['action'], shot['subaction'], shot['camera_id']+1, shot['image_id'])

        input_path = os.path.join(input_dir,subdir)
        output_path = os.path.join(output_dir, subdir)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        input_file_path = os.path.join(input_path, imagename)
        output_file_path = os.path.join(output_path, imagename)

        # process image
        # processed_image,ori_h,ori_w = pre_process(input_file_path, shot['center'], shot['scale'], [192, 256]) #[1, 3, 686, 518]
        image = cv2.imread(input_file_path)
        with torch.no_grad():
            depth = model.infer_image(image) #[1, 1, 686, 518]

        # depth = F.interpolate(depth[None], (ori_h,ori_w), mode='bilinear', align_corners=False)[0, 0] #[256, 192]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        # depth = depth.cpu().numpy().astype(np.uint8)

        cv2.imwrite(output_file_path, depth)
        if idx % 1000 ==0:
            print(f"Processed images are saved in '{imagename}'")

if __name__ == "__main__":
    input_directory = 'H36M-Toolbox/images_crop'
    output_directory = 'H36M-Toolbox/depth_images_v2b'
    
    with open('DCFormer/data/h36m_train.pkl', 'rb') as file:
        train_labels = pickle.load(file)
    file.close()
    # process and save
    process_images_in_directory(train_labels, input_directory, output_directory)

    with open('DCFormer/data/h36m_validation.pkl', 'rb') as file:
        val_labels = pickle.load(file)
    file.close()
    # save process and save
    process_images_in_directory(val_labels, input_directory, output_directory)

    print(f"Processed images are saved in '{output_directory}' with the same structure as '{input_directory}'")

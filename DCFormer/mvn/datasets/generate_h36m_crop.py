import os
from PIL import Image
import numpy as np
import cv2
import pickle
import os

# 定义输入和输出文件夹路径
input_dir = '../H36M-Toolbox/images/'
output_dir = '../H36M-Toolbox/images_crop/'

# 确保输出文件夹存在，如果不存在则创建
os.makedirs(output_dir, exist_ok=True)

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

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    center = np.array(center)
    scale = np.array(scale)

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # rot_rad = np.pi * rot / 180

    # src_dir = get_dir([0, (src_w-1) * -0.5], rot_rad)
    src_dir = np.array([0, (src_w-1) * -0.5], np.float32)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def crop_image(image, center, scale, output_size):
    """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
    Args:
        image numpy array of shape (height, width, 3): input image
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        cropped_image numpy array of shape (height, width, 3): resulting cropped image

    """

    trans = get_affine_transform(center, scale, 0, output_size)
    # inv_trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    image = cv2.warpAffine(
        image,
        trans,
        (output_size),
        flags=cv2.INTER_LINEAR)

    return image, trans

# 遍历输入文件夹中的所有图片文件
def process_images_in_directory(labels):
    joints_2d_gt_crop = []
    inv_trans_all = []
    for idx, shot in enumerate(labels):
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

        image = cv2.imread(
            input_file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        # ori_h,ori_w = image.shape[0],image.shape[1]
        image, trans = crop_image(image, shot['center'], shot['scale'], (192, 256)) # [192, 256]
        cv2.imwrite(output_file_path, image)

if __name__ == "__main__":
    with open('../h36m_validation.pkl', 'rb') as file:
        val_labels = pickle.load(file)
    file.close()
    process_images_in_directory(val_labels)
    del(val_labels)

    with open('../h36m_train.pkl', 'rb') as file:
        train_labels = pickle.load(file)
    file.close()
    process_images_in_directory(train_labels)

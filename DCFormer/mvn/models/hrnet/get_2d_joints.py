import torch
import torch.nn.functional as F
import cv2
import numpy as np
import math
def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: torch.Tensor([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.view((batch_size, num_joints, -1))
    maxvals, idx = torch.max(heatmaps_reshaped, 2)

    maxvals = maxvals.view((batch_size, num_joints, 1))
    idx = idx.view((batch_size, num_joints, 1))

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = (maxvals > 0.0).repeat(1, 1, 2).float()

    preds *= pred_mask
    return preds, maxvals

def get_pred_coords(batch_heatmaps, H, W):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = torch.tensor([hm[py][px + 1] - hm[py][px - 1],
                                     hm[py + 1][px] - hm[py - 1][px]], device=batch_heatmaps.device)
                coords[n][p] += torch.sign(diff) * .25

    orig_coords = coords * torch.tensor([H / heatmap_height, W / heatmap_width], device=batch_heatmaps.device)
    return orig_coords

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

# def exec_affine_transform(pt, t):
#     new_pt = np.array([pt[0], pt[1], 1.]).T
#     new_pt = np.dot(t, new_pt)
#     return new_pt[:2]
def exec_affine_transform(pt, t):
    pt = torch.tensor([pt[0], pt[1], 1.], device=pt.device)  # Torch
    new_pt = torch.matmul(t, pt)
    return new_pt[:2]

def get_ori_coords(preds_2d_joints, inv_trans_all):
    original_keypoints = []
    if preds_2d_joints.dim() == 3:
        preds_2d_joints = preds_2d_joints.unsqueeze(1)  # 添加维度 T
    B,T,J,N=preds_2d_joints.shape
    for b in range(B):
        for t in range(T):
            kp = preds_2d_joints[b, t] #torch.Size([17, 2])
            inv_trans = inv_trans_all[b,t] #torch.Size([2, 3])
            original_kp = torch.zeros_like(kp)
            for j in range(J):
                original_kp[j, 0:2] = exec_affine_transform(kp[j, 0:2], inv_trans)
            original_keypoints.append(original_kp)
    original_keypoints = torch.stack(original_keypoints)
    original_keypoints.view(B, T, J, N)
    return original_keypoints
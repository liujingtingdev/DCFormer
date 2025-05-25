#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math

# from data.reader.img import get_affine_transform, exec_affine_transform

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

def exec_affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):  # heatmap [batch,channel,width,height]
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
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):  # batch
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = exec_affine_transform(coords[p, 0:2], trans)
    return target_coords


def generate_heatmaps(joints, sigma, image_size, heatmap_size, num_joints, **kwargs):
    """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :param sigma:
        :param image_size:
        :param heatmap_size:
        :param num_joints:
        :return: target, target_weight(1: visible, 0: invisible)
    """
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    for index, joint in enumerate(joints):
        x, y = joint
        if x < 0 or y < 0 or x > image_size[0] or y > image_size[1]:
            target_weight[index, 0] = 0

    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = np.array(image_size) / np.array(heatmap_size)
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    # if ("use_different_joints_weight" in kwargs) and (kwargs["use_different_joints_weight"]):
    #     target_weight = np.multiply(target_weight, kwargs["joints_weight"])

    return target, target_weight

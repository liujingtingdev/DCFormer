import torch
import numpy as np
import hashlib
import random
from torch.autograd import Variable
import os
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def loss_depth(keypoints_pred, keypoints_gt, s):
    # Compute regression variance according to:
    # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
    """
    Args:
        keypoints_pred (torch.Tensor): Predicted 3D keypoints (μ), shape (batch_size, K, 1)
        keypoints_gt (torch.Tensor): Ground truth 3D keypoints (J3D), shape (batch_size, K, 1)
        s (torch.Tensor): Predicted uncertainty (log variance s), shape (batch_size, K)

    Returns:
        torch.Tensor: Loss value
    """
    assert keypoints_pred.shape == keypoints_gt.shape

    # This is the log of the variance. We have to clamp it else negative
    # log likelihood goes to infinity.
    # s = torch.clamp(s, -7.0, 7.0)

    # Compute ||J3D - μ||^2
    # loss_depth_reg = 0.5 * torch.exp(-s) * smooth_l1_loss(
    #                 keypoints_pred*10,
    #                 keypoints_gt*10,
    #                 beta=0.0)  # Shape: (batch_size, K)
    diff = (keypoints_pred - keypoints_gt)*100
    print("diff",diff[0])
    print("s",s[0])
    loss_depth_reg = 0.5 * torch.exp(-s) * diff ** 2  # Shape: (batch_size, K)

    loss_covariance_regularize = 0.5 * s
    loss_depth_reg += loss_covariance_regularize

    loss_depth_reg = torch.mean(loss_depth_reg)
    
    return loss_depth_reg


def test_calculation(predicted, target, action, error_sum, data_type, subject, MAE=False):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    if not MAE:
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    batch_num = predicted.size(0)
    frame_num = predicted.size(1)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*batch_num*frame_num, batch_num*frame_num)
    else:
        for i in range(batch_num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item()*frame_num, frame_num)
            
    return action_error_sum


def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
            
    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1':AccumLoss(), 'p2':AccumLoss()} for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
       

def get_varialbe(split, target, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().to(device).view(1, 1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().to(device).view(1, 1, 1, 3)

    joints_left, joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]

    for i in range(len(target)):
        target[i] = target[i].cuda().to(device)

    input_2D, input_2D_crop, img, depth, gt_3D = target

    img = torch.flip(img, [-1])
    img = (img / 255.0 - mean) / std
    depth = depth / 255.0

    if random.random() <= 0.5 and split == "train":
        img = torch.flip(img, [2])
        depth = torch.flip(depth, [2])

        input_2D[:, :, :, 0] *= -1
        input_2D[:, :, joints_left + joints_right] = input_2D[:, :, joints_right + joints_left]

        input_2D_crop[:, :, :, 0] = 192 - input_2D_crop[:, :, :, 0] - 1
        input_2D_crop[:, :, joints_left + joints_right] = input_2D_crop[:, :, joints_right + joints_left]

        gt_3D[:, :, :, 0] *= -1
        gt_3D[:, :, joints_left + joints_right] = gt_3D[:, :, joints_right + joints_left]

    if split == "test":
        img = torch.stack([img, torch.flip(img,[2])], dim=1)
        depth = torch.stack([depth, torch.flip(depth,[2])], dim=1)

        input_2D_flip = input_2D.clone()
        input_2D_flip[:, :, :, 0] *= -1
        input_2D_flip[:, :, joints_left + joints_right] = input_2D_flip[:, :, joints_right + joints_left]
        input_2D = torch.stack([input_2D, input_2D_flip], dim=1)

        input_2D_crop_flip = input_2D_crop.clone()
        input_2D_crop_flip[:, :, :, 0] = 192 - input_2D_crop_flip[:, :, :, 0] - 1
        input_2D_crop_flip[:, :, joints_left + joints_right] = input_2D_crop_flip[:, :, joints_right + joints_left]
        input_2D_crop = torch.stack([input_2D_crop, input_2D_crop_flip], dim=1)

        del input_2D_flip, input_2D_crop_flip

    return [input_2D.float(), input_2D_crop.float(), img.float(), depth.float(), gt_3D.float()]

# def get_varialbe(split, target):
#     num = len(target)
#     var = []
#     if split == 'train':
#         for i in range(num):
#             temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
#             var.append(temp)
#     else:
#         for i in range(num):
#             temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
#             var.append(temp)

#     return var


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")
            
        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
            mean_error_all['p2'].avg))
    
    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def save_model(previous_name, save_dir,epoch, data_threshold, model, model_name):
    # if os.path.exists(previous_name):
    #     os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))

    previous_name = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)
    return previous_name
    
def save_model_new(save_dir,epoch, data_threshold, lr, optimizer, model, model_name):
    # if os.path.exists(previous_name):
    #     os.remove(previous_name)

    # torch.save(model.state_dict(),
    #            '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    torch.save({
                'epoch': epoch,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model.state_dict(),
            },
               '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))

import torch
from torch import nn

from mvn.models import pose_hrnet
from mvn.models.DGLifting import DGLifting

# from mvn.models.depth import DepthAnything
# from depth_anything.dpt import DPT_DINOv2

class DepthGuidedPose(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints

        # HRNet
        if config.model.backbone.type in ['hrnet_32', 'hrnet_48']:
            self.backbone = pose_hrnet.get_pose_net(config.model.backbone)

        # depth model
        # self.depth_anything = DepthAnything()
        # self.depth_anything = DPT_DINOv2()

        if config.model.backbone.fix_weights:
            print("model backbone weights are fixed")
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.Lifting_net = DGLifting(config.model.poseformer)


    def forward(self, images, keypoints_2d_cpn, keypoints_2d_cpn_crop, depth_images):
        device = keypoints_2d_cpn.device
        images = images.permute(0, 3, 1, 2).contiguous() #image:[128, 3, 256, 192],depth:[b, 3, 686, 518]

        keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        # rgb features
        features_list_hr = self.backbone(images) 

        keypoints_3d = self.Lifting_net(keypoints_2d_cpn, keypoints_2d_cpn_crop, depth_images, features_list_hr)

        return keypoints_3d


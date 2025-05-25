from ..backbones.hrnet import HRNet

def __init__(self, cfg, phase, **kwargs):
    self.rough_pose_estimation_net = HRNet(cfg, phase)
def forward(self, x, **kwargs):
    # 单帧
    rough_x = self.rough_pose_estimation_net(target_image)
    # 多帧
    rough_heatmaps = self.rough_pose_estimation_net(torch.cat(x.split(num_color_channels, dim=1), 0))
def init_weights(self):
    if self.freeze_hrnet_weights:
        self.rough_pose_estimation_net.freeze_weight()

concat_input = torch.cat((input_x, input_sup_A, input_sup_B), 1).cuda()
outputs = model(concat_input, margin=margin)
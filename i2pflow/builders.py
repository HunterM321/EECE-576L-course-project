from i2pflow.flow.pose_space import PoseSpace


def build_space(cfg):
    return PoseSpace(cfg.pose.dof, cfg.pose.trans_scale)


def build_head(cfg, space):
    if cfg.method == "flow":
        from i2pflow.models.flow_head import FlowPoseHead
        return FlowPoseHead(cfg, space)
    if cfg.method == "gaussian":
        from i2pflow.models.gaussian_head import GaussianPoseHead
        return GaussianPoseHead(cfg, space)
    raise ValueError(f"unknown head method {cfg.method!r}")


def build_backbone(cfg, opt, device):
    from i2pflow.models.backbone import FrozenCoFiI2P
    return FrozenCoFiI2P(opt, cfg.backbone.ckpt, device)

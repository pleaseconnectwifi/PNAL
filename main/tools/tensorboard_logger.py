# Test git sync
import time
import os
import os.path as osp
import torch
import numpy as np

from .metric_logger import AverageMeter
from tensorboardX import SummaryWriter

_KEYWORDS = ("loss", "pct", "center", "size", "pose")


class TensorboardLogger(object):
    def __init__(self, log_dir, keywords=_KEYWORDS):
        self.log_dir = osp.join(log_dir, "events.{}".format(time.strftime("%m_%d_%H_%M_%S")))
        os.makedirs(self.log_dir, exist_ok=True)
        self.keywords = keywords
        # print(f'Tensorboard wirte to {log_dir}')
        self.writer = SummaryWriter(log_dir=self.log_dir)

        camera_config = {
            'cls': 'PerspectiveCamera',
            'fov': 75,
            'aspect': 0.9,
        }
        self.default_config_dict = {"camera": camera_config}

    def add_scalars(self, meters, step, prefix=""):
        for k, meter in meters.items():
            for keyword in _KEYWORDS:
                if keyword in k:
                    if isinstance(meter, AverageMeter):
                        v = meter.global_avg
                    elif isinstance(meter, (int, float)):
                        v = meter
                    elif isinstance(meter, torch.Tensor):
                        v = meter.cpu().item()
                    else:
                        raise TypeError()

                    self.writer.add_scalar(osp.join(prefix, k), v, global_step=step)

    def add_image(self, img, step, prefix="", tag="img"):
        assert len(img.size()) == 3
        self.writer.add_image(osp.join(prefix, tag),
                              img,
                              global_step=step)

    def add_images(self, batch_img, step, prefix="", tag="batch_img"):
        self.writer.add_images(osp.join(prefix, tag),
                               batch_img,
                               global_step=step)

    def add_mesh(self, vertices, colors, faces, step, prefix="", tag="mesh", config_dict=None):
        """
        Notes:
            convert colors from 0.0 - 1.0 range to 0-255

        """
        if np.max(colors) < 1.01:
            tb_colors = colors * 255
            tb_colors = tb_colors.astype(np.uint8)
        if not config_dict:
            config_dict = self.default_config_dict
        self.writer.add_mesh(f'{tag}',
                             vertices=vertices,
                             colors=tb_colors,
                             global_step=step,
                             config_dict=config_dict,
                             faces=faces)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

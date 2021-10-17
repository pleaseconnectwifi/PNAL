"""
TODO:
    - file logger
"""
from pathlib import Path

from utils.io_utils import IO
from main.viewer.tool import html_from_template


def file_logger_pc(data_batch,
                   preds,
                   output_dir,
                   obj_name,
                   template_fn,
                   mode='test'):

    if mode == 'test':
        # output batch_id = 0
        out_obj_dir = Path(output_dir)/obj_name
        out_obj_dir.mkdir(parents=True, exist_ok=True)
        input_pc = data_batch['pts'][0].cpu().numpy()
        gt_pc = data_batch['gt_pts'][0].cpu().numpy()
        pred_pc = preds['pts'][0].cpu().numpy()

        input_pc_fn = Path(out_obj_dir)/'input.pcd'
        pred_pc_fn = Path(out_obj_dir)/'pred.pcd'
        gt_pc_fn = Path(out_obj_dir)/'gt.pcd'

        IO.put(str(input_pc_fn), input_pc)
        IO.put(str(pred_pc_fn), pred_pc)
        IO.put(str(gt_pc_fn), gt_pc)

        out_html_fn = Path(out_obj_dir, f'{obj_name}.html')
        html_from_template(template_fn,
                           out_html_fn,
                           shape_id=1,
                           input_fn='./input.pcd',
                           pred_fn='./pred.pcd',
                           gt_fn='./gt.pcd')

    else:
        pass





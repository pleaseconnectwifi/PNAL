"""Build network, loss, metric modules"""

from main.tools.file_logger import file_logger_pc
# from main.model import build_PointRefineNet
# from main.dataset_pc import build_pc_loader


# def build_model(cfg, mode):
#     return build_PointRefineNet(cfg, mode)


# def build_data_loader(cfg, mode):
#     return build_pc_loader(cfg, mode)


def build_file_logger(cfg, mode):
    return file_logger_pc

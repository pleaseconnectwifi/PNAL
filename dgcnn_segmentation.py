import os.path as osp
from pointnet2_classification import MLP
from tqdm import tqdm
import argparse
import os,sys,numpy

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import S3DIS
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv
from catalyst.contrib.nn.criterion.ce import SymmetricCrossEntropyLoss


from utils.TruncatedLoss import TruncatedLoss # Generalized Cross Entropy Loss
from main.config import load_cfg_from_file
from main.tools.logger import setup_logger, get_logger
from main.tools.checkpoint import Checkpointer
from main.tools.tensorboard_logger import TensorboardLogger



def parse_args():
    parser = argparse.ArgumentParser(description="VoxelPoint Training")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--which_gpu",default=-1,type=int,)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def preprocess():
    args = parse_args()
    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    exp_name = cfg.DATA.EXP_NAME
    
    log_file_dir = osp.join(osp.dirname(osp.realpath(__file__)),exp_name)
    os.makedirs(log_file_dir, exist_ok=True)
    tensorboard_logger = TensorboardLogger(log_file_dir)
    setup_logger(exp_name, log_file_dir, prefix="train") # name,save_dir,prefix
    logger = get_logger(ext='train')
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))



    # loader for S3DIS
    data_path = cfg.DATA.PC.TRAIN.INPUT_DIR


    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    if cfg.DATA.DATASET == "S3DIS":
        train_dataset = S3DIS(root=data_path,train=True, transform=transform) # WARNING:root:The `pre_transform` argument differs from the one used in the pre-processed version of this dataset. If you really want to make use of another pre-processing technique, make sure to delete `/home/shuquan/hd/shuquan/NL_S3DIS/processed` first.
        test_dataset = S3DIS(root=data_path,train=False, transform=transform)

    if cfg.DATA.SHUFFLE != 0:
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                num_workers=6)
    else: 
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=6)
    return cfg,args,train_dataset,test_dataset,train_loader,test_loader,logger,tensorboard_logger




class Net(torch.nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super(Net, self).__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 9, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Lin(128, out_channels))

    def forward(self, data, need_log_softmax=True):
        x, pos, batch = data.x, data.pos, data.batch
        x0 = torch.cat([x, pos], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)
        if need_log_softmax:
            return F.log_softmax(out, dim=1)
        else:
            return out

def prepare_model(cfg,args,train_dataset,logger):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.DATA.NET == "DGCNN":
        model = Net(train_dataset.num_classes, k=30).to(device)
    else:
        print('cfg.DATA.NET should be one of DGCNN,')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # build checkpointer
    output_dir = osp.join(osp.dirname(osp.realpath(__file__)),cfg.DATA.EXP_NAME)
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                logger=logger)
    checkpoint_data = checkpointer.load(None if cfg.MODEL.WEIGHT=='' else cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    return device,model,optimizer,scheduler,checkpointer,checkpoint_data,ckpt_period

def prepare_loss(cfg,epoch,trainset_size=-1,device=None):
    if cfg.MODEL.LOSS_FUNCTION == '':
        lossf = None
        need_log_softmax = True
    elif cfg.MODEL.LOSS_FUNCTION == 'SCE':
        lossf = SymmetricCrossEntropyLoss(alpha=1.0,beta=1.0)
        need_log_softmax=False
    elif cfg.MODEL.LOSS_FUNCTION == 'GCE':
        assert cfg.DATA.SHUFFLE<=0 # cannot shuffle, need indexes
        lossf = TruncatedLoss(trainset_size=trainset_size).to(device)
        need_log_softmax=False
    elif cfg.MODEL.LOSS_FUNCTION == 'SCE+CE':
        if epoch <= 6:
            lossf = None
            need_log_softmax = True
        else:
            lossf = SymmetricCrossEntropyLoss(alpha=1.0,beta=1.0)
            need_log_softmax=False
    return lossf,need_log_softmax


def train(curr_epoch,train_loader,model,logger,tensorboard_logger,device,lossf=None,need_log_softmax=True):
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    sample_cnt = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data,need_log_softmax)
        if lossf == None:
            loss = F.nll_loss(out, data.y) # log_softmax + nll_loss = cross_entropy
        else:
            if hasattr(lossf,'weight'): # GCE loss
                point_num = out.shape[0]
                indexes = torch.from_numpy(numpy.array(list(range(sample_cnt,sample_cnt+point_num)))).to(device)
                sample_cnt += point_num
                loss = lossf(out, data.y, indexes)
            else: 
                loss = lossf(out, data.y)
        loss.backward()
        optimizer.step()
        if hasattr(lossf,'weight') and curr_epoch%3==2: # GCE loss
            lossf.update_weight(out, data.y, indexes)
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            L = f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} Train Acc: {correct_nodes / total_nodes:.4f}\n'
            logger.info(L)
            loss_dict = {'loss':total_loss}
            metric_dict = {'pct-acc':correct_nodes / total_nodes}
            tensorboard_logger.add_scalars(loss_dict,
                                           curr_epoch * len(train_loader) + i,
                                           prefix="train")
            tensorboard_logger.add_scalars(metric_dict,
                                           curr_epoch * len(train_loader) + i,
                                           prefix="train")
            tensorboard_logger.flush()
            total_loss = correct_nodes = total_nodes = 0


@torch.no_grad()
def test_OA(curr_epoch,loader,model,logger,tensorboard_logger):
    model.eval()
    correct_nodes = total_nodes = 0
    for data in tqdm(loader):
        data = data.to(device)
        pred = model(data)
        correct_nodes += pred.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
    oa = correct_nodes / total_nodes
    L = 'Epoch: {:02d}, Test Acc: {:.4f}\n'.format(epoch, oa)
    logger.info(L)
    metric_dict = {'pct-acc':oa}
    tensorboard_logger.add_scalars(metric_dict,curr_epoch,prefix="test")
    tensorboard_logger.flush()
    return oa


if __name__ == "__main__":
    cfg,args,train_dataset,test_dataset,train_loader,test_loader,logger,tensorboard_logger = preprocess()
    device,model,optimizer,scheduler,checkpointer,checkpoint_data,ckpt_period = prepare_model(cfg,args,train_dataset,logger)

    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = checkpoint_data.get("epoch", 1)
    best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
    best_metric = checkpoint_data.get(best_metric_name, None)
    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in tqdm(range(start_epoch, max_epoch)):
        lossf,need_log_softmax=prepare_loss(cfg,epoch,trainset_size=4096*len(train_dataset),device=device)
        train(epoch,train_loader,model,logger,tensorboard_logger,device,lossf,need_log_softmax)
        oa = test_OA(epoch,test_loader,model,logger,tensorboard_logger)
        if best_metric is None or oa > best_metric:
            best_metric = oa
            checkpoint_data["epoch"] = epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_best", **checkpoint_data)
        # checkpoint
        if epoch % ckpt_period == 1 or epoch == max_epoch:
            checkpoint_data["epoch"] = epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:03d}".format(epoch), **checkpoint_data)
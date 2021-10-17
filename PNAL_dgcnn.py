from pointnet2_classification import MLP
from tqdm import tqdm
import argparse
import os, sys
import os.path as osp
import numpy
import random
import pickle

import torch
def set_random_seed(seed): 
    random.seed(seed) 
    numpy.random.seed(seed) 
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.cuda.manual_seed(seed) 
    return
set_random_seed(0)

import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import S3DIS
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv
from torch.utils.data.sampler import Sampler,BatchSampler


from main.config import load_cfg_from_file
from main.tools.logger import setup_logger, get_logger
from main.tools.checkpoint import Checkpointer
from main.tools.tensorboard_logger import TensorboardLogger
from main.config import load_cfg_from_file

from method import self_correcter
from method import exponential_moving_average




def parse_args():
    parser = argparse.ArgumentParser(description="Training")
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
        train_dataset = S3DIS(root=data_path,train=True, transform=transform) # len 20291 WARNING:root:The `pre_transform` argument differs from the one used in the pre-processed version of this dataset. If you really want to make use of another pre-processing technique, make sure to delete `/processed` first.
        test_dataset = S3DIS(root=data_path,train=False, transform=transform)
    elif cfg.DATA.DATASET == 'SCANNETV2':
        train_dataset = Individual_ScannetV2(root=data_path,train=True, transform=transform, data_dir_name=cfg.DATA.DATA_DIR_NAME)
        test_dataset = ScannetV2(root=data_path,train=False, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=6)

    return cfg,args,train_dataset,test_dataset,test_loader,logger,tensorboard_logger


class NoneBatchPatcher(object):
    def __init__(self,):
        self.corrected=False
        self.last_corrected_label=0


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

def prepare_model(cfg,args,num_classes,logger):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.DATA.NET == "DGCNN":
        model = Net(num_classes, k=30).to(device)
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

    if cfg.DATA != 0:
        ema = exponential_moving_average.EMA(model, decay=cfg.DATA.EMA_DECAY) 
    else:
        ema=None

    return device,model,optimizer,scheduler,checkpointer,checkpoint_data,ckpt_period,ema

def prepare_loss(cfg,epoch):
    if cfg.MODEL.LOSS_FUNCTION == '':
        lossf = None
        need_log_softmax = True
    
    return lossf,need_log_softmax


def save_loss(i,loss,curr_epoch):
    save_path = osp.join(osp.dirname(osp.realpath(__file__)),cfg.DATA.EXP_NAME,str(curr_epoch)+'losses_each_samples.txt')
    f=open(save_path,'ab')
    numpy.savetxt(f,[str(i)+'-th iter'],fmt='%s')
    numpy.savetxt(f,loss.detach().cpu().numpy(),fmt='%1.3f',newline=' ')
    f.close()
    



def train(curr_epoch,train_loader,model,logger,tensorboard_logger,device,lossf=None,need_log_softmax=True,cleaningstage=False,correcter=None,noise_rate=0.6,batch_size=0,perm=None,has_inst=True,ema=None,log_loss=False,ema_lossarray=False):
    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        # if i > 30:
        #     break
        data = data.to(device)
        point_num = data.y.shape[0]
        if perm == None:
            ids = torch.from_numpy(numpy.array(list(range(i*point_num,(i+1)*(point_num))))).to(device)
        else: # shuffled
            ids = torch.from_numpy(numpy.repeat(numpy.array(perm[i])*4096,4096, axis=0)+numpy.array(list(range(4096))*len(perm[i]))) # repeat for all points in cloud and add biases for all points
            ids = ids.to(device)

            if has_inst:
                data_map = torch.cat([data.x, data.pos, torch.unsqueeze(data.y.float(),1), torch.unsqueeze(data.z.float(),1)], dim=1)
            else:
                data_map = torch.cat([data.x, data.pos, torch.unsqueeze(data.y.float(),1)], dim=1)
            ids, indices = torch.sort(ids)
            data_map=data_map[indices]
            if has_inst:
                data.x, data.pos, data.y, data.z = data_map[:,0:6],data_map[:,6:9],data_map[:,9].long(),data_map[:,10].long()
            else:
                data.x, data.pos, data.y = data_map[:,0:6],data_map[:,6:9],data_map[:,9].long()

        if cleaningstage:
            with torch.no_grad():
                if ema_lossarray and (ema is not None):
                    ema.ema.eval() 
                    out = ema.ema(data,need_log_softmax)
                else:
                    model.eval()
                    out = model(data,need_log_softmax)
                predicted_labels = out.argmax(dim=1)
                if lossf == None:
                    loss = F.nll_loss(out, data.y, reduction='none') # log_softmax + nll_loss = cross_entropy
                else:
                    loss = lossf(out, data.y, reduction='none')

            images=torch.cat([data.x, data.pos], dim=-1).clone() #  N, 9
            labels = data.y.clone() # N
            loss_array = loss.clone() # N

            if has_inst:
                _,new_images, new_labels, bp_mask = correcter.threshold_votinpatch_clean_with_reliable_sample_batchg(ids,images, labels, loss_array, noise_rate,predicted_labels,inst=data.z.clone()) # N, 9  N  N 
            else:
                _,new_images, new_labels, bp_mask = correcter.threshold_votinpatch_clean_with_reliable_sample_batchg(ids,images, labels, loss_array, noise_rate,predicted_labels) 

            data.y=new_labels.long()
            data.x, data.pos=new_images[:,:6], new_images[:,6:]



        model.train()

        optimizer.zero_grad()

        out = model(data,need_log_softmax)
        if cleaningstage or log_loss:
            if lossf == None:
                loss = F.nll_loss(out, data.y, reduction='none') # log_softmax + nll_loss = cross_entropy
            else:
                loss = lossf(out, data.y, reduction='none')

            if log_loss:
                save_loss(i,loss,curr_epoch)
                loss = loss.mean()
            else:
                loss = loss.masked_select(bp_mask).mean()
        else:
            if lossf == None:
                loss = F.nll_loss(out, data.y) # log_softmax + nll_loss = cross_entropy
            else:
                loss = lossf(out, data.y)

        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model) 
        total_loss += loss.item()
        predicted_labels = out.argmax(dim=1)
        correct_nodes += predicted_labels.eq(data.y).sum().item()
        total_nodes += data.num_nodes

        # asynchronous history update
        predicted_labels = predicted_labels.clone().detach().cpu() # N
        correcter.async_update_prediction_matrix(ids.cpu(), predicted_labels)


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
def test_OA(curr_epoch,loader,model,logger,tensorboard_logger,ema=None):

    if ema is not None: 
        ema.ema.eval()  
    correct_nodes = total_nodes = 0
    for data in tqdm(loader):
        data = data.to(device)
        if ema is not None:
            pred = ema.ema(data)
        correct_nodes += pred.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
        # break

    oa = correct_nodes / total_nodes
    L = 'Epoch: {:02d}, Test Acc: {:.4f}\n'.format(epoch, oa)
    logger.info(L)
    metric_dict = {'pct-acc':oa}
    tensorboard_logger.add_scalars(metric_dict,curr_epoch,prefix="test")
    tensorboard_logger.flush()
    return oa


def save_correcter(path,correcter):
    save_path = osp.join(osp.dirname(osp.realpath(__file__)),cfg.DATA.EXP_NAME,path)
    checkpoint_data = {}
    checkpoint_data['all_predictions'] = correcter.all_predictions.long().numpy()
    checkpoint_data['corrected_labels'] = correcter.corrected_labels.long().numpy()
    checkpoint_data['update_counters'] = correcter.update_counters.astype(int)
    pickle.dump(checkpoint_data,open(save_path+'.npy','wb'),protocol = 4)

def load_correcter(path,correcter,read_corrected_labels_and_counters=True):
    save_path = osp.join(osp.dirname(osp.realpath(__file__)),cfg.DATA.EXP_NAME,path)
    checkpoint_data=pickle.load(open(save_path+'.npy','rb') ) 


    correcter.all_predictions = torch.from_numpy(checkpoint_data['all_predictions'])
    read_corrected_labels_and_counters = True
    if read_corrected_labels_and_counters:
        correcter.corrected_labels = torch.from_numpy(checkpoint_data['corrected_labels'])
        correcter.update_counters = checkpoint_data['update_counters']
    correcter.type_clear()
    return correcter

def save_ema(path,ema):
    if ema is not None:
        print('saving ema module...')
        save_path = osp.join(osp.dirname(osp.realpath(__file__)),cfg.DATA.EXP_NAME,path+".pkl")
        checkpoint_data = {}
        checkpoint_data["ema"] = ema.state_dict() 
        torch.save(checkpoint_data, save_path)


def load_ema(path,ema):
    print('loading ema module...')
    save_path = osp.join(osp.dirname(osp.realpath(__file__)),cfg.DATA.EXP_NAME,path+".pkl")
    ema.load_checkpoint(save_path) 

    return ema



class RandomSampler(Sampler):
  def __init__(self, prem):
      self.prem=prem


  def __iter__(self):
    return iter(self.prem)

  def __len__(self):
    return len(self.prem)

def shuffle_train_dataset(cfg,train_dataset):
    train_dataset_shuffled,prem = train_dataset.copy().shuffle(return_perm=True)
    sampler = BatchSampler(RandomSampler(prem), batch_size=cfg.TRAIN.BATCH_SIZE,drop_last=False) # drop_last=False default in torch.utils.data.DataLoader
    prem = list(iter(sampler))

    train_loader = DataLoader(train_dataset_shuffled, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=16,pin_memory=True)
    return train_loader,prem # train_dataset is not shuffled!, train_loader is shuffled!


if __name__ == "__main__":
    cfg,args,train_dataset,test_dataset,test_loader,logger,tensorboard_logger = preprocess() # train_dataset is not shuffled!, train_loader is shuffled!

    device,model,optimizer,scheduler,checkpointer,checkpoint_data,ckpt_period,ema = prepare_model(cfg,args,train_dataset.num_classes,logger)

    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = checkpoint_data.get("epoch", 1)
    best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
    best_metric = checkpoint_data.get(best_metric_name, None)

    if cfg.DATA.DATASET == "S3DIS":
        num_points_each_pointcloud = 4096
        batch_size = cfg.TRAIN.BATCH_SIZE*num_points_each_pointcloud 
        num_train_images = (16*1691)*num_points_each_pointcloud 
        num_label = train_dataset.num_classes # 13 in S3DIS
        queue_size = cfg.DATA.QUEUE_SIZE # 4 by default
    elif cfg.DATA.DATASET == "SCANNETV2":
        num_points_each_pointcloud = 4096
        batch_size = cfg.TRAIN.BATCH_SIZE*num_points_each_pointcloud 
        num_train_images = (61778)*num_points_each_pointcloud 
        num_label = 20
        queue_size = cfg.DATA.QUEUE_SIZE 
    threshold = cfg.DATA.THRESHOLD if cfg.DATA.THRESHOLD>0 else 0.05
    L = 'threshold is: {:.4f}, queue_size is: {:02d}\n'.format(threshold, queue_size)
    logger.info(L)
    

    print('Init Correcter ...')
    correcter = self_correcter.Correcter(num_train_images, num_label, queue_size, threshold, loaded_data=[NoneBatchPatcher()]*num_train_images, voting=(cfg.DATA.VOTE!=0), threshold_voting=int(cfg.DATA.THRESHOLD_VOTING),p_not_update=cfg.DATA.P_NOTUPDATE)
    try:
        print('try load latest {:03d}th-epoch correcter ...'.format(start_epoch))
        correcter = load_correcter("model_{:03d}".format(start_epoch),correcter)
    except:
        try:
            print('try load best correcter ...')
            correcter = load_correcter("model_best",correcter)
        except:
            print('====================warning====================\n no correcter is loaded')

    print('Loading EMA ...')
    try:
        print('try load latest {:03d}th-epoch ema ...'.format(start_epoch))
        ema = load_ema("ema_{:03d}".format(start_epoch),ema)
    except:
        print('====================warning====================\n no ema is loaded')
    


    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in tqdm(range(start_epoch, max_epoch)): # epoch from 01 to 30
        lossf,need_log_softmax=prepare_loss(cfg,epoch)
        if cfg.DATA.SHUFFLE != 0 and epoch>0:
            train_loader,perm = shuffle_train_dataset(cfg,train_dataset.copy()) # train_dataset is not shuffled!, train_loader is shuffled!
        else:
            perm=None
            train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=16,pin_memory=True)

        log_loss = cfg.DATA.LOG_LOSS_EACH_SAMPLE>0
        if epoch<=cfg.DATA.WARM_UP: # warm-up periods
            train(epoch,train_loader,model,logger,tensorboard_logger,device,lossf,need_log_softmax,correcter=correcter,batch_size=batch_size,perm=perm,has_inst=cfg.DATA.HAS_INST,ema=ema,log_loss=log_loss,ema_lossarray=cfg.DATA.EMA_LOSSARRAY)
        else: # cleaning stage
            correcter.voting = epoch>=cfg.DATA.VOTE_BEGIN if cfg.DATA.VOTE_BEGIN!=-1 else False
            train(epoch,train_loader,model,logger,tensorboard_logger,device,lossf,need_log_softmax,cleaningstage=True,correcter=correcter,noise_rate=cfg.DATA.NOISE_RATE,batch_size=batch_size,perm=perm,has_inst=cfg.DATA.HAS_INST,ema=ema,log_loss=log_loss,ema_lossarray=cfg.DATA.EMA_LOSSARRAY)
        

        oa = test_OA(epoch,test_loader,model,logger,tensorboard_logger,ema=ema)
        if best_metric is None or oa > best_metric:
            save_correcter("model_best",correcter)
            save_ema("ema_best",ema)
            best_metric = oa
            checkpoint_data["epoch"] = epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_best", **checkpoint_data)
        # checkpoint
        if epoch % ckpt_period == 1 or epoch == max_epoch:
            save_correcter("model_{:03d}".format(epoch),correcter)
            save_ema("ema_{:03d}".format(epoch),ema)
            checkpoint_data["epoch"] = epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:03d}".format(epoch), **checkpoint_data)
        del train_loader,perm

    correcter.predictions_clear()

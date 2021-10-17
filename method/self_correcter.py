import numpy as np
import operator
import torch
from tqdm import tqdm
from structure.minibatch import *
from structure.sample import *

def set_random_seed(seed): 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.cuda.manual_seed(seed) 
    return
set_random_seed(0)


class Correcter(object):
    def __init__(self, size_of_data, num_of_classes, history_length, threshold, loaded_data=None, voting=False, threshold_voting=4,p_not_update=0.0):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.history_length = history_length
        self.threshold = threshold
        self.voting = voting
        self.find_refurbishuncleans_like_selfie = False # we dont need
        self.update_by_clean_labels = False # we dont need
        self.keep_loss_history = False # we dont need
        self.mapdict = {'loss_array':0,'labels':1,'images':[2,-2],'inst':-2,'predicted_labels':-1}
        self.threshold_voting = threshold_voting
        self.p_not_update = p_not_update

        # prediction histories of samples
        self.all_predictions = torch.zeros(size_of_data,history_length).long()-1

        # Max predictive uncertainty
        self.max_certainty = -np.log(1.0/float(self.num_of_classes))

        # Corrected label map
        self.corrected_labels = torch.zeros(size_of_data).long()-1

        self.update_counters = np.zeros(size_of_data, dtype=int)

        if self.keep_loss_history:
            self.loss_history_update_position = 0
            self.loss_history = None

        # For Logging
        self.loaded_data = None
        if loaded_data is not None:
            self.loaded_data = loaded_data
            print('len(self.loaded_data): ',len(self.loaded_data)) 


    def async_update_prediction_matrix(self, ids, predicted_labels):
        # append the predicted label to the prediction matrix
        cur_index = np.array(self.update_counters[ids] % self.history_length)
        self.all_predictions[ids,cur_index] = predicted_labels
        self.update_counters[ids] += 1




    def threshold_votinpatch_clean_with_reliable_sample_batchg(self, ids, images, labels, loss_array, noise_rate, predicted_labels, inst=None): 
        # 1. separate clean and unclean samples
        # print(images.shape, labels.shape, loss_array.shape, noise_rate, predicted_labels.shape) # torch.Size([49152, 9]) torch.Size([49152]) torch.Size([49152]) 0.6 torch.Size([49152])
        if inst is None:
            loss_map = torch.cat([torch.unsqueeze(loss_array,1),torch.unsqueeze(labels,1).float(),images,torch.zeros(len(images),1).to(images.device),torch.unsqueeze(predicted_labels.float(),1)],dim=1) # N,13
        else:
            loss_map = torch.cat([torch.unsqueeze(loss_array,1),torch.unsqueeze(labels,1).float(),images,torch.unsqueeze(inst,1).float(),torch.unsqueeze(predicted_labels.float(),1)],dim=1) # N,13
        # sort loss by descending order
        _, indices = torch.sort(loss_map[:,self.mapdict['loss_array']])
        ids=ids[indices] 
        loss_map=loss_map[indices] 
        
        if not self.find_refurbishuncleans_like_selfie: 
            uncleaned_ids = ids
            uncleaned_batch = loss_map



        # 2. get reliable samples
        # check predictive uncertainty
        pred_history = self.all_predictions[uncleaned_ids].to(predicted_labels.device) # uncleans in all_predictions,  N,4
        most_labels = torch.zeros(pred_history.shape[0],self.num_of_classes)
        for this_class in range(self.num_of_classes):
            most_labels[:,this_class] = torch.sum(pred_history==this_class,dim=1)
        most_labels = torch.unsqueeze(torch.argmax(most_labels,dim=1).to(predicted_labels.device),1) # N,1
        p_dict=(pred_history==most_labels).sum(1) 
        p_dict=p_dict.float()/float(self.history_length)
        # compute predictive uncertainty
        negative_entropy = p_dict * torch.log(p_dict)
        uncertainty = - negative_entropy / self.max_certainty
        # check reliable condition
        reliable_mask = uncertainty<=self.threshold
        rsum = reliable_mask.sum()
        if rsum > 0:
            # update corrected_labels
            if inst is None:
                self.corrected_labels[uncleaned_ids[reliable_mask]] = torch.squeeze(most_labels[reliable_mask].clone().cpu().long(),1)
            else:
                # update by the instance
                corrected_labels_list = torch.squeeze(most_labels[reliable_mask].clone().cpu().long(),1)
                rel_sample_inst_list = []
                if not self.voting:
                    for i,rel_sample_inst in enumerate(uncleaned_batch[:,self.mapdict['inst']][reliable_mask].clone().cpu()):
                        if rel_sample_inst in rel_sample_inst_list:
                            continue
                        else:
                            rel_sample_inst_list.append(rel_sample_inst)
                            inst_mask = uncleaned_batch[:,self.mapdict['inst']]==rel_sample_inst
                            to_correct=self.corrected_labels[uncleaned_ids[inst_mask]]
                            self.corrected_labels[uncleaned_ids[inst_mask]]=corrected_labels_list[i].repeat(to_correct.shape)
                else:
                    rel_sample_inst_list = torch.unique(uncleaned_batch[:,self.mapdict['inst']][reliable_mask].clone().cpu())
                    for i,rel_sample_inst in enumerate(rel_sample_inst_list):
                        # voting among reliable samples
                        inst_mask = uncleaned_batch[:,self.mapdict['inst']]==rel_sample_inst # when correcting, find all samples in this inst
                        ref_inst_mask = uncleaned_batch[:,self.mapdict['inst']][reliable_mask]==rel_sample_inst # when voting, only counts for reliable samples in this inst

                        corrected_labels_of_this_inst = corrected_labels_list[ref_inst_mask]

                        occurances = np.bincount(corrected_labels_of_this_inst.numpy())
                        threshold_occurances_for_voting = max(occurances)//self.threshold_voting
                        winners = np.squeeze(np.argwhere(occurances >= threshold_occurances_for_voting),axis=1) # maybe more than one labels are winner

                        to_correct=self.corrected_labels[uncleaned_ids[inst_mask]].clone()
                        winner_label = np.random.choice(winners,1) if len(winners)>0 else winners
                        origional_label = uncleaned_batch[:,self.mapdict['labels']][inst_mask].clone().cpu().long()

                        print('to_correct_in_corrector: ',to_correct,'winner_label:',winner_label,'origional_label',origional_label)
                        self.corrected_labels[uncleaned_ids[inst_mask]]=torch.from_numpy(winner_label).long().repeat(to_correct.shape)




            # update corrected_batch
            corrected_batch = uncleaned_batch.clone()
            if not self.find_refurbishuncleans_like_selfie:
                # firstly, we set all uncleans label to -1.
                corrected_batch[:,self.mapdict['labels']] = torch.zeros((len(ids))).to(uncleaned_batch.device).float()-1 # all -1
                # secondly, update by corrected labels
                corrected_mask = (self.corrected_labels[uncleaned_ids].to(most_labels.device).long()!=-1)
                corrected_batch_temp = corrected_batch[:,self.mapdict['labels']].clone()
                corrected_batch_temp[corrected_mask] = self.corrected_labels[uncleaned_ids][corrected_mask].to(most_labels.device).float().clone()
                corrected_batch[:,self.mapdict['labels']] = corrected_batch_temp.clone()
                del corrected_batch_temp
            else:
                corrected_batch[:,self.mapdict['labels']]=self.corrected_labels[uncleaned_ids].to(most_labels.device).float().clone()
            not_cleaned_batch = corrected_batch[corrected_batch[:,self.mapdict['labels']].long()==-1]
        else:
            corrected_batch = None
            not_cleaned_batch = uncleaned_batch.clone()

        notr_num = not_cleaned_batch.shape[0]
        not_cleaned_batch[:,self.mapdict['images'][0]:self.mapdict['images'][1]] = torch.zeros((notr_num,9)) # change coord and xyz to 000
        not_cleaned_batch[:,self.mapdict['labels']] = self.num_of_classes # change label to not exist 13
        not_cleaned_batch = not_cleaned_batch.to(most_labels.device)
        print('not_cleaned_batch length: ',not_cleaned_batch.shape[0])
        if rsum > 0:
            corrected_batch[corrected_batch[:,self.mapdict['labels']].long()==-1] = not_cleaned_batch # important!!! or else the not_refurbishable parts in corrected_batch are not updated!
        if not self.find_refurbishuncleans_like_selfie: 
            final_batch = corrected_batch if corrected_batch is not None else not_cleaned_batch

        # sort final batch by ids
        ids, indices = torch.sort(ids)
        final_batch=final_batch[indices]

        not_corrected_mask = final_batch[:,self.mapdict['labels']]==self.num_of_classes
        bp_mask = torch.logical_not(not_corrected_mask)
        temp_final_batch=final_batch[not_corrected_mask]
        temp_final_batch[:,self.mapdict['labels']]=0 # just for loss cal, in which the assigned class does not affect bp and final loss. a rand class >=0 && <self.num_of_classes
        final_batch[not_corrected_mask] = temp_final_batch
        bp_mask = bp_mask.to(most_labels.device)
        ids, images, labels = ids, final_batch[:,self.mapdict['images'][0]:self.mapdict['images'][1]], final_batch[:,self.mapdict['labels']].long()
        assert loss_map.shape == final_batch.shape
        return ids, images, labels, bp_mask
        




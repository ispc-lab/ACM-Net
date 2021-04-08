import os 
import json 
import time
import pickle
from tqdm import tqdm 

import torch 
import wandb 
import numpy as np 
from torch.utils.data import DataLoader

from config.model_config import build_args 
from dataset.dataset_class import build_dataset
from model.ACMNet import ACMNet
from utils.net_utils import set_random_seed, ACMLoss
from utils.net_evaluation import ANETDetection, upgrade_resolution, get_proposal_oic, nms, result2json


"""   
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#                                              TRAIN FUNCTION                                                   #
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
"""
def train(args, model, dataloader, criterion, optimizer):
    model.train()
    print("-------------------------------------------------------------------------------")
    device = args.device
    
    # train_process
    train_num_correct = 0
    train_num_total = 0
    
    loss_stack = []
    act_inst_loss_stack = []
    act_cont_loss_stack = []
    act_back_loss_stack = []
    guide_loss_stack = []
    att_loss_stack = []
    feat_loss_stack = []
    
    for input_feature, vid_label_t in tqdm(dataloader):

        vid_label_t = vid_label_t.to(device)
        input_feature = input_feature.to(device)
        
        act_inst_cls, act_cont_cls, act_back_cls,\
        act_inst_feat, act_cont_feat, act_back_feat,\
        temp_att, act_inst_cas, _, _, _= model(input_feature)
        
        loss, loss_dict = criterion(act_inst_cls, act_cont_cls, act_back_cls, vid_label_t, temp_att,\
                                    act_inst_feat, act_cont_feat, act_back_feat, act_inst_cas)  
        
        optimizer.zero_grad()
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            fg_score = act_inst_cls[:, :args.action_cls_num]
            label_np = vid_label_t.cpu().numpy()
            score_np = fg_score.cpu().numpy()
            
            pred_np = np.zeros_like(score_np)
            pred_np[score_np >= args.cls_threshold] = 1
            pred_np[score_np < args.cls_threshold] = 0
            correct_pred = np.sum(label_np == pred_np, axis=1)
            
            train_num_correct += np.sum((correct_pred == args.action_cls_num))
            train_num_total += correct_pred.shape[0]
            
            loss_stack.append(loss.cpu().item())
            act_inst_loss_stack.append(loss_dict["act_inst_loss"])
            act_cont_loss_stack.append(loss_dict["act_cont_loss"])
            act_back_loss_stack.append(loss_dict["act_back_loss"])
            
            guide_loss_stack.append(loss_dict["guide_loss"])
            feat_loss_stack.append(loss_dict["feat_loss"])
            att_loss_stack.append(loss_dict["sparse_loss"])
            
    train_acc = train_num_correct/train_num_total

    train_log_dict = {}
    train_log_dict["train_act_inst_cls_loss"] = np.mean(act_inst_loss_stack)
    train_log_dict["train_act_cont_cls_loss"] = np.mean(act_cont_loss_stack)
    train_log_dict["train_act_back_cls_loss"] = np.mean(act_back_loss_stack)
    train_log_dict["train_guide_loss"] = np.mean(guide_loss_stack)
    train_log_dict["train_feat_loss"] = np.mean(feat_loss_stack)
    train_log_dict["train_att_loss"] = np.mean(att_loss_stack)
    train_log_dict["train_loss"] = np.mean(loss_stack)
    train_log_dict["train_acc"] = train_acc
    
    print("")
    print("train_act_inst_cls_loss:{:.3f}  train_act_cont_cls_loss:{:.3f}".format(np.mean(act_inst_loss_stack), np.mean(act_cont_loss_stack)))
    print("train_act_back_cls_loss:{:.3f}  train_att_loss:{:.3f}".format(np.mean(act_back_loss_stack), np.mean(att_loss_stack)))
    print("train_feat_loss:        {:.3f}  train_loss:{:.3f}".format(np.mean(feat_loss_stack), np.mean(loss_stack)))
    print("train acc:{:.3f}".format(train_acc))
    print("-------------------------------------------------------------------------------")
    
    return train_log_dict

"""   
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#                                              TEST FUNCTION                                                    #
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
"""

def test(args, model, dataloader, criterion, phase="test"):
    
    model.eval()
    print("-------------------------------------------------------------------------------")
    device = args.device
    save_dir = args.save_dir
    
    test_num_correct = 0
    test_num_total = 0
    
    loss_stack = []
    act_inst_loss_stack = []
    act_cont_loss_stack = []
    act_back_loss_stack = []
    guide_loss_stack = []
    att_loss_stack = []
    feat_loss_stack = []
    
    test_final_result = dict()
    test_final_result['version'] = 'VERSION 1.3'
    test_final_result['results'] = {}
    test_final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}
    
    test_pred_score_stack = []
    test_vid_label_stack = []
    test_tmp_data_log_dict = {}
    
    for vid_name, input_feature, vid_label_t, vid_len, vid_duration in tqdm(dataloader):
        
        input_feature = input_feature.to(device)
        vid_label_t = vid_label_t.to(device)
        vid_len = vid_len[0].cpu().numpy()
        t_factor = (args.segment_frames_num * vid_len) / (args.frames_per_sec * args.test_upgrade_scale  * input_feature.shape[1])
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        act_inst_cls, act_cont_cls, act_back_cls,\
        act_inst_feat, act_cont_feat, act_back_feat,\
        temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas = model(input_feature)
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        loss, loss_dict = criterion(act_inst_cls, act_cont_cls, act_back_cls, vid_label_t, temp_att,\
                                    act_inst_feat, act_cont_feat, act_back_feat, act_inst_cas) 
              
        loss_stack.append(loss.cpu().item())
        act_inst_loss_stack.append(loss_dict["act_inst_loss"])
        act_cont_loss_stack.append(loss_dict["act_cont_loss"])
        act_back_loss_stack.append(loss_dict["act_back_loss"])
        guide_loss_stack.append(loss_dict["guide_loss"])
        att_loss_stack.append(loss_dict["sparse_loss"])
        feat_loss_stack.append(loss_dict["feat_loss"])
        
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        temp_cas = act_inst_cas
        
        test_tmp_data_log_dict[vid_name[0]] = {}
        test_tmp_data_log_dict[vid_name[0]]["vid_len"] = vid_len
        test_tmp_data_log_dict[vid_name[0]]["temp_att_score_np"] = temp_att.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_org_cls_score_np"] = act_cas.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_ins_cls_score_np"] = act_inst_cas.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_con_cls_score_np"] = act_cont_cas.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_bak_cls_score_np"] = act_back_cas.cpu().numpy()
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        fg_score = act_inst_cls[:, :args.action_cls_num]
        label_np = vid_label_t.cpu().numpy()
        score_np = fg_score.cpu().numpy()
        pred_np = np.zeros_like(score_np)
        pred_np[score_np >= args.cls_threshold] = 1
        pred_np[score_np < args.cls_threshold] = 0
        correct_pred = np.sum(label_np == pred_np, axis=1)
        test_num_correct += np.sum((correct_pred == args.action_cls_num))
        test_num_total += correct_pred.shape[0]
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        # GENERATE PROPORALS.
        temp_cls_score_np = temp_cas[:, :, :args.action_cls_num].cpu().numpy()
        temp_cls_score_np = np.reshape(temp_cls_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        temp_att_ins_score_np = temp_att[:, :, 0].unsqueeze(2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
        temp_att_con_score_np = temp_att[:, :, 1].unsqueeze(2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
        temp_att_ins_score_np = np.reshape(temp_att_ins_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        temp_att_con_score_np = np.reshape(temp_att_con_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        
        score_np = np.reshape(score_np, (-1))
        if score_np.max() > args.cls_threshold:
            cls_prediction = np.array(np.where(score_np > args.cls_threshold)[0])
        else:
            cls_prediction = np.array([np.argmax(score_np)], dtype=np.int)
            
        temp_cls_score_np = temp_cls_score_np[:, cls_prediction]
        temp_att_ins_score_np = temp_att_ins_score_np[:, cls_prediction]
        temp_att_con_score_np = temp_att_con_score_np[:, cls_prediction]
        
        
        test_tmp_data_log_dict[vid_name[0]]["temp_cls_score_np"] = temp_cls_score_np
        
        int_temp_cls_scores = upgrade_resolution(temp_cls_score_np, args.test_upgrade_scale)
        int_temp_att_ins_score_np = upgrade_resolution(temp_att_ins_score_np, args.test_upgrade_scale)
        int_temp_att_con_score_np = upgrade_resolution(temp_att_con_score_np, args.test_upgrade_scale) 
        
        
        cas_act_thresh = np.arange(0.15, 0.25, 0.05)
        att_act_thresh = np.arange(0.15, 1.00, 0.05)
        
        proposal_dict = {}
        # CAS based proposal generation
        # cas_act_thresh = []
        for act_thresh in cas_act_thresh:

            tmp_int_cas = int_temp_cls_scores.copy()
            zero_location = np.where(tmp_int_cas < act_thresh)
            tmp_int_cas[zero_location] = 0
            
            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_cas[:, c_idx] >= act_thresh)
                tmp_seg_list.append(pos)
            
            props_list = get_proposal_oic(tmp_seg_list, (1.0*tmp_int_cas + 0.0*int_temp_att_ins_score_np), cls_prediction, score_np, t_factor, lamb=0.2, gamma=0.0)
            
            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]
                
                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []
                
                proposal_dict[class_id] += props_list[i]
        
        # att_act_thresh = []
        for att_thresh in att_act_thresh:

            tmp_int_att = int_temp_att_ins_score_np.copy()
            zero_location = np.where(tmp_int_att < att_thresh)
            tmp_int_att[zero_location] = 0
            
            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_att[:, c_idx] >= att_thresh)
                tmp_seg_list.append(pos)
            
            props_list = get_proposal_oic(tmp_seg_list, (1.0*int_temp_cls_scores + 0.0*tmp_int_att), cls_prediction, score_np, t_factor, lamb=0.2, gamma=0.0)
            
            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]
                
                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []
                
                proposal_dict[class_id] += props_list[i]
        
        # NMS 
        final_proposals = []
        
        for class_id in proposal_dict.keys():
            final_proposals.append(nms(proposal_dict[class_id], args.nms_thresh))
                
        test_final_result['results'][vid_name[0]] = result2json(final_proposals, args.class_name_lst)
        
    test_acc = test_num_correct / test_num_total
    
    if args.test:
        # Final Test
        test_pred_txt_f = os.path.join(save_dir, "final_test_pred.txt")
        test_label_txt_f = os.path.join(save_dir, "final_test_label.txt")
        test_final_json_path = os.path.join(save_dir, "final_test_{}_result.json".format(args.dataset))
    else:
        # Train Evalutaion
        test_pred_txt_f = os.path.join(save_dir, "test_pred.txt")
        test_label_txt_f = os.path.join(save_dir, "test_label.txt")
        test_final_json_path = os.path.join(save_dir, "{}_lateset_result.json".format(args.dataset))
    
    np.savetxt(test_pred_txt_f, np.array(test_pred_score_stack), fmt="%.3f")
    np.savetxt(test_label_txt_f, np.array(test_vid_label_stack), fmt="%.3f")
    
    with open(test_final_json_path, 'w') as f:
        json.dump(test_final_result, f)
    
    anet_detection = ANETDetection(ground_truth_file=args.test_gt_file_path,
                    prediction_file=test_final_json_path,
                    tiou_thresholds=args.tiou_thresholds,
                    subset="test")
    
    test_mAP = anet_detection.evaluate()
    
    print("")
    print("test_act_inst_cls_loss:{:.3f}  test_act_cont_cls_loss:{:.3f}".format(np.mean(act_inst_loss_stack), np.mean(act_cont_loss_stack)))
    print("test_act_back_cls_loss:{:.3f}  test_att_loss:{:.3f}".format(np.mean(act_back_loss_stack), np.mean(att_loss_stack)))
    print("test_feat_norm_loss:   {:.3f}  test_loss:{:.3f}".format(np.mean(feat_loss_stack), np.mean(loss_stack)))
    print("test acc:{:.3f}".format(test_acc))
    print("-------------------------------------------------------------------------------")
    
    test_log_dict = {}
    test_log_dict["test_act_inst_cls_loss"] = np.mean(act_inst_loss_stack)
    test_log_dict["test_act_cont_cls_loss"] = np.mean(act_cont_loss_stack)
    test_log_dict["test_act_back_cls_loss"] = np.mean(act_back_loss_stack)
    test_log_dict["test_feat_loss"] = np.mean(feat_loss_stack)
    test_log_dict["test_att_loss"] = np.mean(att_loss_stack)
    test_log_dict["test_loss"] = np.mean(loss_stack)
    test_log_dict["test_acc"] = test_acc
    test_log_dict["test_mAP"] = test_mAP
        
    return test_log_dict, test_tmp_data_log_dict

"""   
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#                                              MAIN FUNCTION                                                    #
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
"""   
def main(args):
    
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    local_time = time.localtime()[0:5]
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    if not args.test:
        save_dir = os.path.join(this_dir, "checkpoints_acmnet", "checkpoints_acmnet_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}"\
                                        .format(local_time[0], local_time[1], local_time[2],\
                                                local_time[3], local_time[4]))
    else:
        save_dir = os.path.dirname(args.checkpoint)

    args.save_dir = save_dir
    args.device = device
        
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    model = ACMNet(args)
    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    
    if not args.test:
        if not args.without_wandb:
            wandb.init(name='traing_log_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}'\
                            .format(local_time[0], local_time[1], local_time[2],
                                    local_time[3], local_time[4]),
                    config=args,
                    project="ACMNet_{}".format(args.dataset),
                    sync_tensorboard=True)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        train_dataset = build_dataset(args, phase="train", sample="random") 
        test_dataset = build_dataset(args, phase="test", sample="uniform") 
            
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                    num_workers=args.num_workers, drop_last=False)
        
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False)
        
        criterion = ACMLoss(lamb1=args.loss_lamb_1, lamb2=args.loss_lamb_2, lamb3=args.loss_lamb_3)
        
        best_test_mAP = 0
        
        for epoch_idx in tqdm(range(args.start_epoch, args.epochs)):
        
            train_log_dict = train(args, model, train_dataloader, criterion, optimizer)
            
            with torch.no_grad():
                test_log_dict, test_tmp_data_log_dict = test(args, model, test_dataloader, criterion)
                test_mAP = test_log_dict["test_mAP"]
                
            if test_mAP > best_test_mAP:
                best_test_mAP = test_mAP
                checkpoint_file = "{}_best_checkpoint.pth".format(args.dataset)
                torch.save({
                    'epoch':epoch_idx,
                    'model_state_dict':model.state_dict()
                    }, os.path.join(save_dir, checkpoint_file))
                                    
                with open(os.path.join(save_dir, "test_tmp_data_log_dict.pickle"), "wb") as f:
                    pickle.dump(test_tmp_data_log_dict, f)

            checkpoint_file = "{}_latest_checkpoint.pth".format(args.dataset)
            torch.save({
                'epoch':epoch_idx,
                'model_state_dict':model.state_dict()
                }, os.path.join(save_dir, checkpoint_file))
            
            print("Current test_mAP:{:.4f}, Current Best test_mAP:{:.4f} Current Epoch:{}/{}".format(test_mAP, best_test_mAP, epoch_idx, args.epochs))
            print("-------------------------------------------------------------------------------")
            
            if not args.without_wandb:
                wandb.log(train_log_dict)
                wandb.log(test_log_dict)

            # optimizer_scheduler.step()
    else:
        test_dataset = build_dataset(args, phase="test", sample="uniform")
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False)
        criterion = ACMLoss()
        
        with torch.no_grad():
            test_log_dict, test_tmp_data_log_dict = test(args, model, test_dataloader, criterion)
            test_mAP = test_log_dict["test_mAP"]
            
            with open(os.path.join(save_dir, "test_tmp_data_log_dict.pickle"), "wb") as f:
                pickle.dump(test_tmp_data_log_dict, f)
                

if __name__ == "__main__":
    
    set_random_seed()
    args = build_args(dataset="THUMOS")
    print(args)
    main(args)
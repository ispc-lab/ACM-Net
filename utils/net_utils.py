import torch 
import random 
import numpy as np 
import torch.nn as nn 

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
def weights_init(model):
    if isinstance(model, nn.Conv2d):
        model.weights.data.normal_(0.0, 0.001)
    elif isinstance(model, nn.Linear):
        model.weights.data.normal_(0.0, 0.001)

    
class ACMLoss(nn.Module):
    
    def __init__(self, lamb1=2e-3, lamb2=5e-5, lamb3=2e-4, dataset="THUMOS14"):
        super(ACMLoss, self).__init__()
        
        self.dataset = dataset
        self.lamb1 = lamb1 # att_norm_loss param 
        self.lamb2 = lamb2
        self.lamb3 = lamb3 
        self.feat_margin = 50  #50
        
    def cls_criterion(self, inputs, label):
        return - torch.mean(torch.sum(torch.log(inputs) * label, dim=1))
    
    def forward(self, act_inst_cls, act_cont_cls, act_back_cls, vid_label, temp_att=None,\
                act_inst_feat=None, act_cont_feat=None, act_back_feat=None, temp_cas=None):
        
        device = act_inst_cls.device 
        batch_size = act_inst_cls.shape[0]
        
        act_inst_label = torch.hstack((vid_label, torch.zeros((batch_size, 1), device=device)))
        act_cont_label = torch.hstack((vid_label, torch.ones((batch_size, 1), device=device)))
        act_back_label = torch.hstack((torch.zeros_like(vid_label), torch.ones((batch_size, 1), device=device)))
        
        act_inst_label = act_inst_label / torch.sum(act_inst_label, dim=1, keepdim=True)
        act_cont_label = act_cont_label / torch.sum(act_cont_label, dim=1, keepdim=True)
        act_back_label = act_back_label / torch.sum(act_back_label, dim=1, keepdim=True)
        
        act_inst_loss = self.cls_criterion(act_inst_cls, act_inst_label)
        act_cont_loss = self.cls_criterion(act_cont_cls, act_cont_label)
        act_back_loss = self.cls_criterion(act_back_cls, act_back_label)
        
        # Guide Loss
        guide_loss = torch.sum(torch.abs(1 - temp_cas[:, :, -1] - temp_att[:, :, 0].detach()), dim=1).mean()

        # Feat Loss
        act_inst_feat_norm = torch.norm(act_inst_feat, p=2, dim=1)
        act_cont_feat_norm = torch.norm(act_cont_feat, p=2, dim=1)
        act_back_feat_norm = torch.norm(act_back_feat, p=2, dim=1)
        
        feat_loss_1 = self.feat_margin - act_inst_feat_norm + act_cont_feat_norm
        feat_loss_1[feat_loss_1 < 0] = 0
        feat_loss_2 = self.feat_margin - act_cont_feat_norm + act_back_feat_norm
        feat_loss_2[feat_loss_2 < 0] = 0
        feat_loss_3 = act_back_feat_norm
        feat_loss = torch.mean((feat_loss_1 + feat_loss_2 + feat_loss_3)**2)

        # Sparse Att Loss
        # att_loss = torch.sum(temp_att[:, :, 0], dim=1).mean() + torch.sum(temp_att[:, :, 1], dim=1).mean() 
        sparse_loss = torch.sum(temp_att[:, :, :2], dim=1).mean()
        
        if self.dataset == "THUMOS14":
            cls_loss = 1.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            cls_loss = 5.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss
            
        add_loss = self.lamb1 * guide_loss + self.lamb2 * feat_loss + self.lamb3 * sparse_loss
        
        loss = cls_loss + add_loss
        
        loss_dict = {}
        loss_dict["act_inst_loss"] = act_inst_loss.cpu().item()
        loss_dict["act_cont_loss"] = act_cont_loss.cpu().item()
        loss_dict["act_back_loss"] = act_back_loss.cpu().item()
        loss_dict["guide_loss"] = guide_loss.cpu().item()
        loss_dict["feat_loss"] = feat_loss.cpu().item()
        loss_dict["sparse_loss"] = sparse_loss.cpu().item()
        
        return loss, loss_dict
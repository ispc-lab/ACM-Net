import torch 
import torch.nn as nn 

class ACMNet(nn.Module):
    
    def __init__(self, args):
        super(ACMNet, self).__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.action_cls_num = args.action_cls_num # Only the action categories number.
        self.drop_thresh = args.dropout
        self.ins_topk_seg = args.ins_topk_seg 
        self.con_topk_seg = args.con_topk_seg 
        self.bak_topk_seg = args.bak_topk_seg
        
        self.dropout = nn.Dropout(args.dropout)
        if self.dataset == "THUMOS":
            self.feature_embedding = nn.Sequential(
                # nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            self.feature_embedding = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        
        # We introduce three-branch attention, action instance, action context and the irrelevant backgrounds.
        self.att_branch = nn.Conv1d(in_channels=self.feature_dim, out_channels=3, kernel_size=1, padding=0)
        self.snippet_cls = nn.Linear(in_features=self.feature_dim, out_features=(self.action_cls_num + 1))
        
    def forward(self, input_features):

        device = input_features.device
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]
        
        inst_topk_num = max(temp_len // self.ins_topk_seg, 1)
        cont_topk_num = max(temp_len // self.con_topk_seg, 1)
        back_topk_num = max(temp_len // self.bak_topk_seg, 1)
        
        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)
        
        if self.dataset == "THUMOS":
            temp_att = self.att_branch((embeded_feature))
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            temp_att = self.att_branch(self.dropout(embeded_feature))
        
        temp_att = temp_att.permute(0, 2, 1)
        temp_att = torch.softmax(temp_att, dim=2)
        
        act_inst_att = temp_att[:, :, 0].unsqueeze(2)
        act_cont_att = temp_att[:, :, 1].unsqueeze(2)
        act_back_att = temp_att[:, :, 2].unsqueeze(2)

        embeded_feature = embeded_feature.permute(0, 2, 1)
        embeded_feature_rev = embeded_feature
        
        select_idx = torch.ones((batch_size, temp_len, 1), device=device)
        select_idx = self.dropout(select_idx)
        embeded_feature = embeded_feature * select_idx

        act_cas = self.snippet_cls(self.dropout(embeded_feature))
        act_inst_cas = act_cas * act_inst_att
        act_cont_cas = act_cas * act_cont_att
        act_back_cas = act_cas * act_back_att
        
        sorted_inst_cas, _ = torch.sort(act_inst_cas, dim=1, descending=True)
        sorted_cont_cas, _ = torch.sort(act_cont_cas, dim=1, descending=True)
        sorted_back_cas, _ = torch.sort(act_back_cas, dim=1, descending=True)
        
        act_inst_cls = torch.mean(sorted_inst_cas[:, :inst_topk_num, :], dim=1)
        act_cont_cls = torch.mean(sorted_cont_cas[:, :cont_topk_num, :], dim=1)
        act_back_cls = torch.mean(sorted_back_cas[:, :back_topk_num, :], dim=1)
        act_inst_cls = torch.softmax(act_inst_cls, dim=1)
        act_cont_cls = torch.softmax(act_cont_cls, dim=1)
        act_back_cls = torch.softmax(act_back_cls, dim=1)
        
        act_inst_cas = torch.softmax(act_inst_cas, dim=2)
        act_cont_cas = torch.softmax(act_cont_cas, dim=2)
        act_back_cas = torch.softmax(act_back_cas, dim=2)
        
        act_cas = torch.softmax(act_cas, dim=2)
        
        _, sorted_act_inst_att_idx = torch.sort(act_inst_att, dim=1, descending=True)
        _, sorted_act_cont_att_idx = torch.sort(act_cont_att, dim=1, descending=True)
        _, sorted_act_back_att_idx = torch.sort(act_back_att, dim=1, descending=True)
        act_inst_feat_idx = sorted_act_inst_att_idx[:, :inst_topk_num, :].expand([-1, -1, self.feature_dim])
        act_cont_feat_idx = sorted_act_cont_att_idx[:, :cont_topk_num, :].expand([-1, -1, self.feature_dim])
        act_back_feat_idx = sorted_act_back_att_idx[:, :back_topk_num, :].expand([-1, -1, self.feature_dim])
        act_inst_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_inst_feat_idx), dim=1)
        act_cont_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_cont_feat_idx), dim=1)
        act_back_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_back_feat_idx), dim=1)
        
        return act_inst_cls, act_cont_cls, act_back_cls,\
               act_inst_feat, act_cont_feat, act_back_feat,\
               temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas


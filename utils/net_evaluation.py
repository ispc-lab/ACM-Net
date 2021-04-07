import json
from typing import IO 
import numpy as np 
import pandas as pd 
from joblib import Parallel, delayed
import scipy
from sklearn.metrics import average_precision_score
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from torch.utils import data

# ACKNOWLEDGEMENT.
# THIS PART CODE IS CONSTRUCTED BASED ON ACTIVITYNET GIT-HUB REPO.

def get_wtCAM(temp_cls_scores, temp_attention, pred_cls):
    """
    Calculate the temporal weighted classification scores for the pred_cls

    Args:
        tCAM ([T x Cls_Dim]): 
        temporal_attention ([T x 1]): 
        pred_cls ([Cls_Dim]): specify which class we interested, could be multi-hot predicted cls vector
    """
    # temp_att_cls_scores = temp_cls_scores * temp_attention
    # temp_att_cls_scores = (temp_cls_scores + temp_attention) / 2.0
    temp_att_cls_scores = temp_cls_scores
    temp_att_cls_scores = temp_att_cls_scores[:, pred_cls]
    
    return temp_att_cls_scores

def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interpolate.interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def temporal_interpolation(temp_data, scale):
    """
    Temporal Linear Interpolation Func.
    Args:
        temp_data ([T, dim]): the target array needed to be interpolated.
        scale ([int / float]): temporal interpoalted coefficient.

    Returns:
        np.array [T x scale, dim]: temporal intepolated arrays
    """
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    temp_len = temp_data.shape[0]
    temp_x = np.arange(temp_len)
    f = interpolate.interp1d(temp_x, temp_data, axis=0, kind="linear", fill_value="extrapolate")
    int_temp_x = np.arange(0, temp_len, step=1/scale)
    temp_data = f(int_temp_x)
    
    return temp_data

def get_tempseg_list(temp_att_cls_scores, temp_attention, thr=0.1, dataset="THUMOS"):
    """
    Return the index where the temp_att_cls_scores are greater than the threshold
    Args:
        temp_att_cls_scores ([temp_len x c_len]): [temp_attention_weighted action class classification scores]
        temp_attention ([temp_len x 1]): [temp_attention]
        thr (float, optional): [threshod]. Defaults to 0.05.
    """
    temp = []
    c_len = temp_att_cls_scores.shape[1]
    for c_idx in range(c_len):
        if dataset == "THUMOS":
            # the return of np.where is a tuple. such as (array([], dtype=int64), )
            # pos = np.where((0.8 * temp_att_cls_scores[:, c_idx] + 0.2 * temp_attention[:, 0] * temp_attention[:, 0]) > thr)
            pos = np.where(temp_att_cls_scores[:, c_idx] > thr)
        elif dataset == "ActivityNet":
            # The previous work apply a gaussian_filter for the ActivityNet temp_attention_cls_map

            pos = np.where(temp_att_cls_scores[:, c_idx] > thr)
        else:
            raise ValueError("WRONG DATASET when obtaining the tempseg_list")
        
        temp.append(pos)
    
    return temp
    
def grouping(arr):
    """
    Group the connected results
    """
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

def get_temp_proposal(tempseg_list, int_temp_scores, c_pred, c_pred_scores, t_factor):
    """
    Obtaining the action instance porposals.
    
    Args:
        tempseg_list ([type]): [description]
        int_temp_scores ([type]): [description]
        c_pred ([type]): [description]
        t_factor ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    temp_proposal = []
    temp_len = int_temp_scores.shape[0]
    int_temp_scores = int_temp_scores.reshape(temp_len, -1)
    c_len = int_temp_scores.shape[1]

    for c_idx in range(c_len):
        c_temp_proposal = []
        c_temp_seg_list = np.array(tempseg_list[c_idx][0])
        if c_temp_seg_list.any():
            grouped_c_temp_list = grouping(c_temp_seg_list)
            # Apply the Outer-Inner-Contrasive func as score function
            for j in range(len(grouped_c_temp_list)):
                
                if grouped_c_temp_list[j][0] > 0:
                    left_bound = max(0, grouped_c_temp_list[j][0] - len(grouped_c_temp_list[j])//4 - 1)
                    left_outer_idxs = np.arange(left_bound, grouped_c_temp_list[j][0])
                    c_left_outer = np.mean(int_temp_scores[left_outer_idxs, c_idx])
                else:
                    c_left_outer = 0
                
                if grouped_c_temp_list[j][-1] < temp_len - 1:
                    right_bound = min(temp_len, grouped_c_temp_list[j][-1] + len(grouped_c_temp_list[j])//4 + 1)
                    right_outer_idxs = np.arange(grouped_c_temp_list[j][-1], right_bound)
                    c_right_outer = np.mean(int_temp_scores[right_outer_idxs, c_idx])
                else:
                    c_right_outer = 0 
                
                c_scores = np.mean(int_temp_scores[grouped_c_temp_list[j], c_idx]) - (c_left_outer + c_right_outer) / 2 + 0.3 * c_pred_scores[c_idx]
                
                t_start = grouped_c_temp_list[j][0] * t_factor
                t_end = grouped_c_temp_list[j][-1] * t_factor
                # if t_end - t_start < 0.1:
                #     continue
                c_temp_proposal.append([c_pred[c_idx], c_scores, t_start, t_end])

        temp_proposal.append(c_temp_proposal)
        
    return temp_proposal

def get_proposal_oic(tempseg_list, int_temp_scores, c_pred, c_pred_scores, t_factor, lamb=0.25, gamma=0.20): # [0.25, 0.20]
    temp = []
    for i in range(len(tempseg_list)):
        c_temp = []
        temp_list = np.array(tempseg_list[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                if len(grouped_temp_list[j]) < 2:
                    continue
                
                inner_score = np.mean(int_temp_scores[grouped_temp_list[j], i])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - lamb * len_proposal))
                outer_e = min(int(int_temp_scores.shape[0] - 1), int(grouped_temp_list[j][-1] + lamb * len_proposal))

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))
                
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(int_temp_scores[outer_temp_list, i])

                c_score = inner_score - outer_score + gamma * c_pred_scores[c_pred[i]]
                t_start = (grouped_temp_list[j][0]  + 0) * t_factor
                t_end =   (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])
                    
            temp.append(c_temp)
    return temp



def result2json(temp_prop_lst, class_name_lst):
    result = []
    for i in range(len(temp_prop_lst)):
        for j in range(len(temp_prop_lst[i])):
            line = {'label': class_name_lst[int(temp_prop_lst[i][j][0])],
                    'score': temp_prop_lst[i][j][1],
                    'segment': [temp_prop_lst[i][j][2], temp_prop_lst[i][j][3]]}
            result.append(line)

    return result

            
def get_cls_ap(prediction_scores, gt_labels, dataset='THUMOS'):
    """
    Calculate the video-level classification results for weakly-supervised temporal action detection task.

    Args:
        prediction_scores ([N x cls_num]): [video level classification scores]
        gt_labels ([N x cls_num]): [video level ground truth labels] (could be multiple hot.)
    """
    result_ap_lst = []
    # pre_scores = prediction_scores.detach().cpu().numpy()
    # ground_truth = gt_labels.detach().cpu().numpy()
    pre_scores = np.array(prediction_scores)
    ground_truth = np.array(gt_labels)
    
    # -------------------------------------------------------# 
    # Since the CliffDiving is actually a subset of Diving. Follow CDC (CVPR 2017)
    # We resign the classification scores of Diving when CliffDiving scores is higher than Diving
    if dataset == "THUMOS":
        switch_idx = pre_scores[:, 4] > pre_scores[:, 7]
        pre_scores[switch_idx, 7] = pre_scores[switch_idx, 4]

    cls_num = pre_scores.shape[1] - 1 # ignore the background
    for cls_idx in range(cls_num):
        ap = average_precision_score(ground_truth[:, cls_idx].astype(np.int64), pre_scores[:, cls_idx])
        result_ap_lst.append(ap)
        
    cls_mAP = sum(result_ap_lst) / len(result_ap_lst)
    
    return cls_mAP


def get_segment_iou(target_segment, candidate_segment):
    """
    Calculate the t-IOU between target_segments and the candidate_segments.
    
    Args:
        target_segment (1d array): [t_start, t_end]
        candidate_segment (2d array): N X [t_start, t_end]
    Return:
        tIOU
    """
    tt1 = np.maximum(target_segment[0], candidate_segment[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segment[:, 1])
    segment_intersection = (tt2 - tt1).clip(0)
    segment_union = (candidate_segment[:, 1] - candidate_segment[:, 0]) + \
                    (target_segment[1] - target_segment[0]) - segment_intersection
    tIOU = segment_intersection.astype(np.float) / segment_union    
    
    return tIOU

def wrapper_get_segment_iou(target_segments, candidate_segments):
    """Compute temporal iou btw segments

    Args:
        target_segments (2d array): m x [t_start, t_end]
        candidate_segments (2d array): n x [t_start, t_end]
    
    Return:
        tIOU (2d array) [n x m] with iou ratios
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError("Dimension of candidate_segemnts is incorrect!!!")
    
    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for idx in range(m):
        tiou[:, idx] = get_segment_iou(target_segments[idx, :], candidate_segments)
    
    return tiou

def get_AP(prec, rec):
    """
    Calculate the interpolated AP -- VOCdevkit from VOC 2011
    
    Args:
        prec ([type]): [description]
        rec ([type]): [description]

    Returns:
        AP [float]:
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for idx in range(len(mprec) - 1)[::-1]:
        mprec[idx] = max(mprec[idx], mprec[idx + 1])    
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    
    return ap

def minmax_data_norm(data):
    """
    Args:
        data ([N, Cls]):
    """
    max_data = np.max(data, axis=0)
    min_data = np.min(data, axis=0)
    delta = max_data - min_data
    data = (max_data - data) / (max_data - min_data + 1e-4)
    data = np.clip(data, a_min=0.0, a_max=1.0)
    return data


def nms(proposals, thresh):
    
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]

    return keep

class ANETDetection(object):
    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']
    
    def __init__(self, ground_truth_file=None, prediction_file=None,\
                ground_truth_fields=GROUND_TRUTH_FIELDS,\
                prediction_fields=PREDICTION_FIELDS,\
                tiou_thresholds=np.linspace(0.1, 0.9, 9),\
                subset="validation", verbose=True,\
                check_status=True):
        
        if not ground_truth_file:
            raise IOError("No specified ground_truth file.")
        if not prediction_file:
            raise IOError("No specified prediction file.")
        
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        
        self.blocked_videos = []
        
        self.ground_truth, self.activity_index = self._import_ground_truth(ground_truth_file)
        self.prediction = self._import_prediction(prediction_file)
        
        if self.verbose:
            # print("[INIT] Loaded annotations from {} subset.".format(subset))
            print("\n\t Number of ground truth instances: {}".format(len(self.ground_truth)))
            print("\t Number of predictions: {}".format(len(self.prediction)))
            # print("\t Specified Temporal IOU:{}".format(self.tiou_thresholds))
        
        
    def _import_ground_truth(self, ground_truth_file_name):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_file_name : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_file_name, "r") as f:
            data = json.load(f)
        
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground_truth file.')
        
        # Load Ground Truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for video_id, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if video_id in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(video_id)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])
        
        ground_truth = pd.DataFrame({'video-id':video_lst,
                                     't-start':t_start_lst,
                                     't-end':t_end_lst,
                                     'label':label_lst})
        
        print(len(activity_index))
        
        return ground_truth, activity_index
    
        
    def _import_prediction(self, prediction_file_name):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_file_name, 'r') as f:
            predict_data = json.load(f)
            
        if not all([field in predict_data.keys() for field in self.pred_fields]):
            raise IOError("Pelase input a valid prediction file")
    
        video_lst, t_start_lst, t_end_lst, label_lst, score_lst = [], [], [], [], []
        for video_id, v in predict_data['results'].items():
            if video_id in self.blocked_videos:
                continue
            for pred in v:
                label = self.activity_index[pred['label']]
                video_lst.append(video_id)
                t_start_lst.append(float(pred['segment'][0]))
                t_end_lst.append(float(pred['segment'][1]))
                label_lst.append(label)
                score_lst.append(float(pred['score']))
        
        prediction = pd.DataFrame({'video-id':video_lst,
                                   't-start':t_start_lst,
                                   't-end':t_end_lst,
                                   'label':label_lst,
                                   'score':score_lst})
    
        return prediction
    
    
    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            # print("Warning: No predictions of label {} were provided".format(label_name))
            return pd.DataFrame
    
    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')
        
        results = Parallel(n_jobs=10)(delayed(compute_average_precision_detection)(
                                    ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                                    prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                                    tiou_thresholds=self.tiou_thresholds,
                                    ) for label_name, cidx in self.activity_index.items())
        
        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]
        
        return ap
    
    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()
        self.mAP = self.ap.mean(axis=1)
        self.mAP = self.mAP.reshape(len(self.mAP))
        self.average_mAP = self.mAP.mean()
        
        print("-------------------------------------------------------------------------------")
        print('|t-IoU |{}|'.format("||".join(["{:.3f}".format(item) for item in self.tiou_thresholds])))
        print("-------------------------------------------------------------------------------")
        print('|mAP   |{}|'.format("||".join(["{:.3f}".format(item) for item in self.mAP])))
        print("-------------------------------------------------------------------------------")
        print('|Average-mAP: {:.4f}'.format(self.average_mAP))
        print("-------------------------------------------------------------------------------")

        return self.average_mAP
    
    
def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.1, 0.9, 9)):
    """Compute average precision (detection task) between ground truth and predictions data frames.
    If multiple predictions occurs for the same predicted segment, only the one with highest score is
    mathced as positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (df): 
            Data frame containing the ground truth instances.
            Required fields: ['video_id', 't-start', 't-end']
            
        prediction (df): 
            Data frame containing the prediction instances.
            Required fields: ['video-id', 't-start', 't-end', 'score']
        
        tiou_thresholds (1darray, optional):
            Temporal intersection over union threshold.
            Defaults to np.linspace(0.1, 0.9, 9).
            
    Outpus:
    ap: float
        average precision scores.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap
    
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # sort predictions by decreasing score order
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)
    
    # Initializa true positive and false positive vectors
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))
    
    ground_truth_gbvn = ground_truth.groupby('video-id')
    
    for idx, this_pred in prediction.iterrows():
        
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            # wrong predicted association cls label.
            fp[:, idx] = 1
            continue
        
        this_gt = ground_truth_videoid.reset_index()
        
        tiou_arr = get_segment_iou(this_pred[['t-start', 't-end']].values,
                                   this_gt[['t-start', 't-end']].values)
        
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after filters above
                tp[tidx, idx] = 1
                # for each gt, we only assign the highest iou detection instance.
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx

                break
                
            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1 
                
    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    
    recall_cumsum = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = get_AP(precision_cumsum[tidx, :], recall_cumsum[tidx, :])
    
    return ap
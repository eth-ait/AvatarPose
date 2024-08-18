from AvatarPose.vis.vis_pose import get_keypoints3d
from os.path import join
import os
from glob import glob
import numpy as np
from scripts.eval_metrics import metric_mesh, metric_mpjpe, correct_limbs, eval_list_to_ap, eval_list_to_mpjpe, eval_list_to_recall, eval_list_to_precision, eval_list_to_pck


def eval_list_to_mpjpe(eval_list, threshold=500):
    eval_list.sort(key=lambda k: k["score"])
    gt_det = []

    mpjpes = []
    for i, item in enumerate(eval_list):
        if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
            mpjpes.append(item["mpjpe"])
            gt_det.append(item["gt_id"])

    return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf


def metric_mpjpe(predict, target):
    # predict: {'name':(19, 3)}
    # target: (19, 3)
    loss = 500
    final_name = ''
    i = 0
    idx = i
    for name in predict.keys():
        mpjpe = np.sqrt(((predict[name][:, :3] - target[:, :3])**2).sum(axis=-1)).mean() * 1000 # mm
        if mpjpe < loss:
            loss = mpjpe
            final_name = name
            idx = i
        i+=1
    return loss, final_name, idx

def cal_loss(root_1, root_2, seqs_1, seqs_2, kpts_name_1, kpts_name_2, start=None, end=None):
    """
    names_1 and names_2 must be related to the same sequences with the same length
    """
    mpjpe_thresh = 500
    eval_list = []
    total_gt = 0
    total_p_1 = 0
    total_p_2 = 0
    correct_bones = {'human_0':[], 'human_1':[]}
    all_bones = {'human_0':[], 'human_1':[]}
    for i in range(len(seqs_1)):
        seq_1 = seqs_1[i]
        seq_2 = seqs_2[i]
        root_1_joints = join(root_1, seq_1, kpts_name_1)
        root_2_joints = join(root_2, seq_2, kpts_name_2)
        names = glob(join(root_2_joints, '*.json'))
        frames = [int(os.path.basename(name).split('.')[0]) for name in names]
        indexs = np.argsort(frames)
        frames = [frames[idx] for idx in indexs]
        for frame_idx, frame in enumerate(frames):
            if start is not None and frame < start:
                continue
            if end is not None and frame >= end:
                continue
            name_1 = join(root_1_joints, f'{frame:06d}.json')
            name_2 = join(root_2_joints, f'{frame:06d}.json')
            
            # get joints
            dict_1 = get_keypoints3d(name_1)
            dict_2 = get_keypoints3d(name_2)
            
            total_p_1 += len(dict_1)
            total_p_2 += len(dict_2)
            
            
            # find corresponding gt, for calculating mpjpe
            for p in dict_2.keys():
                metric, name, idx = metric_mpjpe(dict_1, dict_2[p])
                if name != '':
                    eval_list.append({
                        'mpjpe': metric,
                        'gt_id': int(total_gt + idx),
                        'score': metric
                    })   
            # fine corresponding pred, for calculaing PCP  
            dict_pred_new = {} # change to name of gt    
            total_gt += len(dict_1)
            for p in dict_1.keys():
                metric, name, idx = metric_mpjpe(dict_2, dict_1[p])
                if name != '':
                    dict_pred_new[p] = dict_2[name][:, :3]
                        
            # calcuate correct limbs
            for p in dict_pred_new.keys():
                correct_limbs_cnt, limbs_cnt = correct_limbs(dict_pred_new[p], dict_1[p])
                correct_bones[p].append(correct_limbs_cnt)
                all_bones[p].append(limbs_cnt)
    recall = eval_list_to_recall(eval_list, total_gt)
    precision = eval_list_to_precision(eval_list)
    mpjpe = eval_list_to_mpjpe(eval_list, mpjpe_thresh)
    ap_dict = {}
    for t in [25, 50, 100, 150]:
        ap, rec = eval_list_to_ap(eval_list, total_gt, t)
        ap_dict[f'AP@{t}'] = ap *100
    
    PCK = eval_list_to_pck(eval_list, thres=50)
    PCP = {}
    for name in correct_bones.keys():
        correct_bones[name] = np.mean(correct_bones[name])
        all_bones[name] = np.mean(all_bones[name])
        PCP[name] = (correct_bones[name] / all_bones[name]) * 100
    PCP_all = 0
    print(PCP)
        
    PCP_avg = np.mean([PCP[name] for name in PCP.keys()])
    result = {
        'mpjpe': mpjpe,
        'PCP_avg': PCP_avg,
        'ap': ap_dict,
        'PCK': PCK,
        'F1': 2 * precision * recall / (precision + recall) * 100
    }
    for key in result.keys():
        print(f'{key}: {result[key]}')
    return result

if __name__ == '__main__':
    from glob import glob
    import os

    root_1 = '/home/username/Hi4D_AvatarPose'
    root_2 = '/home/username/outputs/hi4d/exp'
    

    seqs_1 = ['pair14_talk14']
    seqs_2 = ['14_talk14']

    kpts_name_1 = 'skel19_gt'
    kpts_name_2 = 'kpts_est_kpts_opt_1'

    cal_loss(root_1, root_2, seqs_1, seqs_2, kpts_name_1, kpts_name_2)
    
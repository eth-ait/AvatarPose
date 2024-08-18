import numpy as np

# # skel19 
limbs = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [1, 6], [2, 7], [3, 8], [4, 9], [4, 10],
         [5, 11], [6, 12], [7, 13], [8, 14], [11, 15], [12, 16], [13, 18], [14, 17]]

def metric_mesh(predict, target):
    # predict: (6890, 3)
    # target: (6890, 3)
    final_loss = 500
    final_name = ''
    i = 0
    idx = i
    for name in predict.keys():
        loss = np.sqrt(((predict[name] - target)**2).sum(axis=-1)).mean() * 1000
        if loss < final_loss:
            final_loss = loss
            final_name = name
            idx=i
        i+=1
    return final_loss, final_name, idx

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

def correct_limbs(predict, target, alpha=0.5):
    # predict: (19, 3)
    # target: (19, 3)
    correct_limbs = 0
    total_limbs = 0
    for j, limb in enumerate(limbs):
        total_limbs += 1
        error_s = np.linalg.norm(predict[limb[0], :] - target[limb[0], :])
        error_e = np.linalg.norm(predict[limb[1], :] - target[limb[1], :])
        limb_length = np.linalg.norm(target[limb[0], :] - target[limb[1], :])
        if (error_s + error_e) / 2.0 < alpha * limb_length:
            correct_limbs += 1
            
    # pred_hip = (predict[2, 0:3] + predict[3, 0:3]) / 2.0
    # gt_hip = (target[2] + target[3]) / 2.0
    # total_limbs += 1
    # error_s = np.linalg.norm(pred_hip - gt_hip)
    # error_e = np.linalg.norm(predict[12, 0:3] - target[12])
    # limb_length = np.linalg.norm(gt_hip - target[12])
    # if (error_s + error_e) / 2.0 <= alpha * limb_length:
    #     correct_limbs += 1
    return correct_limbs, total_limbs


def eval_list_to_ap(eval_list, total_gt, threshold):
    eval_list.sort(key=lambda k: k["score"])
    total_num = len(eval_list)

    tp = np.zeros(total_num)
    fp = np.zeros(total_num)
    gt_det = []
    for i, item in enumerate(eval_list):
        if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
            tp[i] = 1
            gt_det.append(item["gt_id"])
        else:
            fp[i] = 1
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (total_gt + 1e-5)
    precise = tp / (tp + fp + 1e-5)
    for n in range(total_num - 2, -1, -1):
        precise[n] = max(precise[n], precise[n + 1])

    precise = np.concatenate(([0], precise, [0]))
    recall = np.concatenate(([0], recall, [1]))
    index = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

    return ap, recall[-2]

def eval_list_to_pck(eval_list, thres=50):
    total_num = len(eval_list)
    corr_kpts = np.zeros(total_num)
    for i, item in enumerate(eval_list):
        if item['mpjpe'] < thres:
            corr_kpts[i] = 1 
    return corr_kpts.mean()

def eval_list_to_mpjpe(eval_list, threshold=500):
    eval_list.sort(key=lambda k: k["score"])
    gt_det = []

    mpjpes = []
    for i, item in enumerate(eval_list):
        if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
            mpjpes.append(item["mpjpe"])
            gt_det.append(item["gt_id"])

    return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

def eval_list_to_recall(eval_list, total_gt, threshold=500):
    gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

    return len(np.unique(gt_ids)) / total_gt

def eval_list_to_recall(eval_list, total_gt, threshold=500):
    gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

    return len(np.unique(gt_ids)) / total_gt

def eval_list_to_precision(eval_list, threshold=500):
    eval_list.sort(key=lambda k: k["score"])
    total_num = len(eval_list)

    tp = np.zeros(total_num)
    fp = np.zeros(total_num)
    gt_det = []
    for i, item in enumerate(eval_list):
        if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
            tp[i] = 1
            gt_det.append(item["gt_id"])
        else:
            fp[i] = 1
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    precise = tp[-1] / (tp[-1] + fp[-1])
    return precise
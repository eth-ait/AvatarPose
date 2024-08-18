import numpy as np
from AvatarPose.utils.cam_utils import read_cameras
from AvatarPose.utils.file_utils import write_keypoints3d, read_json
import json
from os.path import join
import os
import cv2
from glob import glob
from tqdm import tqdm
# body25 pairs and paf indices
pose_body_part_pairs = [1,8, 9,10,  10,11,  8,9,  8,12,  12,13, 13,14,  1,2,  2,3,  3,4,   2,17, 1,5,    5,6,   6,7,  5,18,  1,0,  0,15,  0,16, 15,17,  16,18,  14,19,19,20,14,21, 11,22,22,23,11,24]
pose_map_index = [0,1, 2,3, 4,5, 6,7,  8,9,  10,11, 12,13,  14,15, 16,17,  18,19, 20,21, 22,23, 24,25, 26,27, 28,29,  30,31, 32,33, 34,35, 36,37,  38,39,   40,41,42,43,44,45, 46,47,48,49,50,51]
pose_pairs = np.array(pose_body_part_pairs).reshape(-1, 2)
map_idx = np.array(pose_map_index).reshape(-1, 2)

def loss_keypoints3d(predict, target):
    loss = 1e8
    final_pid = 0
    predict = np.array(predict)
    for k in range(len(target)):
        pid = target[k]['id']
        target_kpts = np.array(target[k]['keypoints3d'])
        mpjpe = np.sqrt(((predict - target_kpts)**2).sum(axis=-1)).mean() * 1000 # mm
        if mpjpe < loss:
            loss = mpjpe
            final_pid = pid
    return loss, final_pid

def read_cands_dict(json_root, frame):
    json_name = join(json_root, f'{frame:06d}_keypoints.json')
    f = open(json_name, 'r')
    data = json.load(f)
    cands_dict = data['part_candidates'][0]
    for key, val in cands_dict.items():
        cands_dict[key] = np.array(val).reshape(-1, 3)
    return cands_dict

def read_pafs(paf_root, frame):
    paf_name = join(paf_root, f'{frame:06d}_pose_heatmaps.float')
    x = np.fromfile(paf_name, dtype=np.float32)
    assert x[0] == 3 
    shape_x = x[1:1+int(x[0])]
    assert len(shape_x) == 3 
    H_net = shape_x[1]
    W_net = shape_x[2]
    arrayData = x[1+int(round(x[0])):]
    pafs = arrayData.reshape(shape_x.astype(int))
    return pafs, W_net, H_net

class fDAssociation:
    def __init__(self, 
                 datas_root = './Hi4D_AvatarPose', 
                 openpose_root = './openpose/openpose',
                 data_name = 'pair14_hug14'):
        self.datas_root = datas_root
        self.data_name = data_name
        self.openpose_root = openpose_root
        
        self.paf_score_th = 0.05
        self.conf_th = 0.95
        self.paf_root = 'output_heatmaps_folder'
        self.json_root = 'output_jsons'
        self.data4d_root = './third_parties/4d_association/data/seq'
        self.fDassociation_root = './third_parties/4d_association/build/linux-native'
        self.project_root = os.getcwd()
        self.detec_root = join(self.project_root, f'outputs/openpose/seq', self.data_name)
        self.data_root = join(self.datas_root, self.data_name)
        self.image_root = join(self.data_root, 'images')
        self.out_root = join(self.data_root, 'skel19_4DA')
        self.gt_root = join(self.data_root, 'skel19_gt')
        os.makedirs(self.data4d_root, exist_ok=True)
        
        cameras_root = sorted(glob(join(self.image_root, '*')))
        self.cameras = [os.path.basename(camera_root) for camera_root in cameras_root]
        images = sorted(glob(join(cameras_root[0], '*.jpg')))
        if len(images) == 0:
            images = sorted(glob(join(cameras_root[0], '*.png')))
        self.img_size = [cv2.imread(images[0]).shape[1], cv2.imread(images[0]).shape[0]]
        self.frames = self.get_frames()
        
    def get_frames(self):
        image_names = glob(os.path.join(self.image_root, list(self.cameras)[0], '*'))
        frames = [int(os.path.basename(name).split('.')[0]) for name in image_names]
        start_frame = min(frames)
        end_frame = max(frames)
        frames = np.arange(start_frame, end_frame+1, 1)
        return frames
        
    def save_cam_json(self):
        # save camera calibration matrix
        cams_dict = read_cameras(self.data_root)
        dict_new = {}
        
        for key in cams_dict.keys():
            if key not in self.cameras:
                continue
            dict_new[key] = {}
            for k in cams_dict[key].keys():
                if k == 'K':
                    dict_new[key][k] = cams_dict[key][k].reshape(-1).tolist()
                elif k == 'R':
                    dict_new[key][k] = cams_dict[key][k].reshape(-1).tolist()
                elif k == 'T':
                    dict_new[key][k] = cams_dict[key][k].reshape(-1).tolist()
            dict_new[key]['imgSize'] = self.img_size
            
        with open(join(self.data4d_root, "calibration.json"), "w") as outfile:
            json.dump(dict_new, outfile)
        print("calibration.json saved")
    
    def save_videos_avi(self):
        # save videos
        os.makedirs(join(self.data4d_root, 'video'), exist_ok=True)

        for cam in self.cameras:
            images = sorted(glob(join(self.image_root, cam, '*.jpg')))
            frame_width, frame_height = cv2.imread(images[0]).shape[1], cv2.imread(images[0]).shape[0]
            
            videoname = join(self.data4d_root, f'video/{cam}.avi')
            video_writer = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*"MJPG"), 30, (frame_width, frame_height))
            for image in tqdm(images):
                video_writer.write(cv2.imread(image))
        print("video/cam.avi saved")
                
    
    def save_openpose_output(self):
        os.chdir(self.openpose_root)
        for cam in self.cameras:
            image_dir = join(self.image_root, cam)
            paf_dir = join(self.detec_root, cam, self.paf_root)
            json_dir = join(self.detec_root, cam, self.json_root)
            os.makedirs(paf_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)
            cmd = f'./build/examples/openpose/openpose.bin --image_dir {image_dir} --heatmaps_add_PAFs --write_heatmaps_format float --write_heatmaps {paf_dir} --write_json {json_dir} --keypoint_scale 3 --heatmaps_scale 0 --part_candidates True --display 0 --render_pose 0'
            os.system(cmd)
        os.chdir(self.project_root)
        print("paf and json saved")
    
    # def save_gt_txt(self):  
    #     # save gt keypoints
    #     keypoints_generator = SMPLKeypoints(root = self.data_root,
    #                                         smpl_root = 'smpl_gt/smpl.npz',
    #                                         pids = [0, 1],
    #                                         frames = self.frames,
    #                                         out_root = self.data4d_root,
    #                                         out_name = 'gt.txt')
    #     keypoints_generator.save_keypoints(
    #         func = lambda pid, frame_idx: keypoints_generator.get_model_keypoints(pid, frame_idx, model_type='shelf15'), 
    #         out_type = 'txt')
        
    #     print("gt.txt saved")
    
    def save_openpose_txt(self):      
        # save openpose detections
        n_joints = 25

        os.makedirs(join(self.data4d_root, 'detection'), exist_ok=True)
        for cam in self.cameras:
            paf_dir = join(self.detec_root, cam, self.paf_root)
            json_dir = join(self.detec_root, cam, self.json_root)
            with open(f'{self.data4d_root}/detection/{cam}.txt', 'w') as f:
                f.write(str(4))
                f.write('\t')
                f.write(str(len(self.frames)))
                f.write('\n')
                for frame in tqdm(self.frames):
                    # read pafs, keypoint candidates
                    pafs, W_net, H_net = read_pafs(paf_dir, frame)
                    cands_dict = read_cands_dict(json_dir, frame)
                    threshold = np.sqrt(W_net*H_net) / 150
                    # write candidate keypoints
                    for j in range(n_joints):
                        cands = cands_dict[str(j)]
                        f.write(str(cands.shape[0]))
                        f.write('\n')
                        for i in range(cands.shape[1]):
                            for j in range(cands.shape[0]):
                                f.write(str(cands[j, i]))
                                f.write('\t')
                            f.write('\n')
                
                    for l in range(len(pose_pairs)):
                        # limb number
                        pafX = pafs[map_idx[l][0], :, :]
                        pafY = pafs[map_idx[l][1], :, :]
                        cands1 = cands_dict[str(pose_pairs[l][0])]
                        cands2 = cands_dict[str(pose_pairs[l][1])]
                        for i in range(cands1.shape[0]):
                            for j in range(cands2.shape[0]):
                                # different candidate pairs
                                cand1 = cands1[i, :2].copy()
                                cand2 = cands2[j, :2].copy()
                                cand1[0] = min(W_net-1, max(0, int(round(cand1[0] * W_net)))) # what is the exact coordinate ?
                                cand2[0] = min(W_net-1, max(0, int(round(cand2[0] * W_net))))
                                cand1[1] = min(H_net-1, max(0, int(round(cand1[1] * H_net))))
                                cand2[1] = min(H_net-1, max(0, int(round(cand2[1] * H_net))))
                                d_ij = np.subtract(cand2, cand1)
                                d_max = max(abs(d_ij))
                                n_interp_samples = max(5, min(25, int(round(5*np.sqrt(d_max)))))
                                
                                norm = np.linalg.norm(d_ij)
                                
                                if norm:
                                    d_ij = d_ij / norm
                                    
                                if norm != 0:
                                    # interpolate between the two joints
                                    interp_coord = list(zip(np.linspace(cand1[0], cand2[0], num=n_interp_samples),
                                                    np.linspace(cand1[1], cand2[1], num=n_interp_samples)))
                                    interp_coord = np.array(interp_coord)
                                    interp_coord[:, 0] = np.round(interp_coord[:, 0]).astype(int)
                                    interp_coord[:, 1] = np.round(interp_coord[:, 1]).astype(int)
                                    # the interpolated paf values
                                    paf_interp = []
                                    for k in range(len(interp_coord)):
                                        paf_interp.append([pafX[int(interp_coord[k][1]), int(interp_coord[k][0])], pafY[int(interp_coord[k][1]), int(interp_coord[k][0])]])

                                    paf_scores = np.dot(paf_interp, d_ij)
                                    mask = np.where(paf_scores > self.paf_score_th)[0]
                                    ratio = len(mask) / n_interp_samples
                                    avg_paf_score = sum(paf_scores[mask]) / n_interp_samples
                                    if ratio > self.conf_th :
                                        score = avg_paf_score ### ?
                                    elif norm < threshold:
                                        score = 0.05 + 1e-6
                                    else:
                                        score = 0
                                else:
                                    score = 0 
                            
                                f.write(str(score))
                                f.write('\t')
                            f.write('\n')
        print("openpose detection/cam.txt saved")
                            
    def run_4Dassociation(self):
        # run 4DAssociation
        os.chdir(self.fDassociation_root)
        os.system('./four_d_association/evaluate_shelf')
        os.chdir(self.project_root)
        print("skel.txt saved")

    def keypoints_from_txt(self):
        # extract dectected keypoints from txt and save to body25 json format
        skel_name = join(self.data4d_root, 'skel.txt')
        file = open(skel_name).readlines()

        # read from txt to the format [[{'id': , 'keypoints3d': }, {'id': , 'keypoints3d': }], ...], len = n_frames
        values = file[0].split('\n')[0].split('\t')
        n_joints = int(values[0])
        n_frames = int(values[1])
        assert(n_frames == len(self.frames))

        keypoints_list_all = []
        line = 1
        for i in range(n_frames):
            n_humans = int(file[line].split('\n')[0])
            line += 1
            keypoints_list = []
            for j in range(n_humans):
                pid = int(file[line].split('\n')[0])
                keypoints = []
                line += 1
                for k in range(4):
                    values = file[line].split('\n')[0].split(' ')
                    values = [float(val) for val in values if val != '']
                    keypoints.append(values)
                    line += 1
                keypoints = np.array(keypoints).reshape(4, n_joints).T
                keypoints_dict = {'id': pid, 'keypoints3d': keypoints[:, :3]} #/keypoints[:, 3:]
                keypoints_list.append(keypoints_dict)
            write_keypoints3d(join(self.out_root, f'{self.frames[i]:06d}.json'), keypoints_list)
            keypoints_list_all.append(keypoints_list)
        
        print("skel19_4DA/frame:06d.json saved")
            
    def adjust_pid_with_keypoints(self):
        for i in range(len(self.frames)):
            keypoints_init = read_json(join(self.out_root, f'{self.frames[i]:06d}.json'))
            keypoints_gt = read_json(join(self.gt_root, f'{self.frames[i]:06d}.json'))
            if len(keypoints_init) == len(keypoints_gt):
                break
        print("detect all people from frame", self.frames[i])
        # find the best match
        pid_dict = {}
        for j in range(len(keypoints_init)):
            pid_init = keypoints_init[j]['id']
            keypoints_init_pid = keypoints_init[j]['keypoints3d']
            loss, pid_gt = loss_keypoints3d(keypoints_init_pid, keypoints_gt)
            pid_dict[pid_init] = pid_gt
        
        for i in range(len(self.frames)):
            keypoints_init = read_json(join(self.out_root, f'{self.frames[i]:06d}.json'))
            # adjust the pid
            for j in range(len(keypoints_init)):
                keypoints_init[j]['id'] = pid_dict[keypoints_init[j]['id']]
                keypoints_init[j]['keypoints3d'] = np.array(keypoints_init[j]['keypoints3d'])
            write_keypoints3d(join(self.out_root, f'{self.frames[i]:06d}.json'), keypoints_init)
        
        # smpl_init_all = dict(np.load(join(self.data_root, 'smpl_init', 'smpl.npz'), allow_pickle=True))
        # new_smpl = {}
        # for key in smpl_init_all.keys():
        #     pid = int(key.split('_')[-1])
        #     name = f'human_{pid_dict[pid]}'
        #     new_smpl[key] = smpl_init_all[name]
        # np.savez(join(self.data_root, 'smpl_init', 'smpl.npz'), **new_smpl)
            
        # for i in range(len(self.frames)):
        #     smpl_init = read_json(join(self.data_root, 'smpl_init', f'{self.frames[i]:06d}.json'))
        #     for j in range(len(smpl_init)):
        #         smpl_init[j]['id'] = pid_dict[smpl_init[j]['id']]
        #         for key in smpl_init[j].keys():
        #             if key != 'id':
        #                 smpl_init[j][key] = np.array(smpl_init[j][key])
        #     write_smpl(join(self.data_root, 'smpl_init', f'{self.frames[i]:06d}.json'), smpl_init)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--openpose', type=str, required=True)
    args = parser.parse_args()
    

    print("****processing ", args.seq, "****")
    fd = fDAssociation(
                datas_root = args.data_root, 
                openpose_root = args.openpose,
                data_name = args.seq)
    fd.save_cam_json() # calibration.json
    fd.save_videos_avi() # video/cam.avi
    fd.save_openpose_output() # run openpose save paf, json
    fd.save_openpose_txt() # detection/cam.txt 
    # fd.save_gt_txt() # gt.txtpp
    fd.run_4Dassociation() # skel.txt
    fd.keypoints_from_txt() # skel19_4DA/frame:06d.json
    os.system('rm -r outputs/openpose')
    os.system('rm -r third_parties/4d_association/data/seq')
    # fd.adjust_pid_with_keypoints()
'''
modified from SNARF
'''
import torch
import numpy as np
from instant_avatar.deformers.smpl.body_models import SMPL
import cv2
from scipy.spatial.transform import Rotation
import pickle
import hydra


class SMPLServer(torch.nn.Module):

    def __init__(self, pid, gender='neutral', betas=None):
        super().__init__()

        self.pid = pid
        self.smpl = SMPL(model_path=hydra.utils.to_absolute_path('./data/SMPLX/smpl'),
                         gender=gender,
                         batch_size=1,
                         use_hands=False,
                         use_feet_keypoints=False,
                         dtype=torch.float32)#.cuda() # might use: self.smpl.lbs_weights

        if betas is not None:
            self.betas = torch.from_numpy(betas).reshape(1, -1)
        else:
            print('SMPL Server should specify the person identity with the shape(beta) information')
            raise ValueError

        self.bone_parents = self.smpl.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        for i in range(24): 
            self.bone_ids.append([self.bone_parents[i], i])

        self.c = self.canonical()
        self.c_verts = self.c['c_verts']
        self.c_tfs = self.c['c_tfs']

        Tpose = self.Tpose()
        self.T_verts = Tpose['smpl_verts'] 
        self.t_Jtrs = Tpose['smpl_Jtrs']

        self.cc = self.canonical_centered()
    
    def Tpose(self):
        poses_t = np.zeros((1, 69), dtype=np.float32)
        Rh_t = np.zeros((1, 3), dtype=np.float32)
        Th_t = np.zeros((1, 3), dtype=np.float32)
        output = self.forward(poses_t, Rh_t, Th_t)
        return output
    
    def t_Jtrs_norm(self, cc=None):
        if cc is None:
            cc_coord_max = self.cc['cc_coord_max']
            cc_coord_min = self.cc['cc_coord_min']
            tn_Jtrs = self.t_Jtrs-self.cc['c_center']
        else:
            cc_coord_max = cc['cc_coord_max']
            cc_coord_min = cc['cc_coord_min']
            tn_Jtrs = self.t_Jtrs-cc['c_center']
        padding = (cc_coord_max - cc_coord_min) * 0.05
        tn_Jtrs = (tn_Jtrs-cc_coord_min+padding)/(cc_coord_max - cc_coord_min) / 1.1 ##???
        tn_Jtrs -= 0.5
        tn_Jtrs *= 2 # ???
        return tn_Jtrs
    
    def minimal_shape(self, pose):
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))
        pose_mat = pose.as_matrix()       # 23 x 3 x 3
        # Minimally clothed shape
        # data_n = pickle.load(open('smpl/smpl_model/SMPL_NEUTRAL.pkl', 'rb'), encoding='latin1')
        # posedir = data_n['posedirs']
        # print(posedir.dtype)
        posedir = self.smpl.posedirs

        ident = np.eye(3)
        pose_feature = (pose_mat - ident).reshape([207, 1])
        pose_offsets = np.dot(posedir.reshape([-1, 207]), pose_feature).reshape([6890, 3])
        minimal_shape = self.T_verts.numpy() + pose_offsets
        T = np.matmul(self.smpl.lbs_weights, self.c_tfs.reshape([-1, 16])).reshape([-1, 4, 4])
        minimal_verts = np.matmul(T[:, :3, :3],   minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        return minimal_verts


    def canonical(self):
        # define the canonical pose(A pose)
        poses_c = np.zeros((1, 69), dtype=np.float32)#.cuda()
        poses_c[0, 2] = np.pi/4
        poses_c[0, 5] = -np.pi / 4
        Rh_c = np.zeros((1, 3), dtype=np.float32)
        Th_c = np.zeros((1, 3), dtype=np.float32)#.cuda()
        
        smpl_output = self.forward(poses_c, Rh_c, Th_c)
        # get same data from the canonical pose
        output = {
            'c_verts': smpl_output['smpl_verts'],
            'c_jnts': smpl_output['smpl_jnts'],
            'c_tfs': smpl_output['smpl_tfs'].squeeze(0),
            'c_tfs_inv': smpl_output['smpl_tfs'].squeeze(0).inverse(),
            'c_Jtrs': smpl_output['smpl_Jtrs'] # here Joints are in T pose, not A pose
        }
        return output

    def canonical_centered(self, minimal=False, minimal_verts=None):
        if not minimal:
            c_center = torch.mean(self.c_verts.squeeze(0), dim=0)
            cc_verts = self.c_verts-c_center
        else:
            c_center = np.mean(minimal_verts, axis=0)
            cc_verts = minimal_verts-c_center
        cc_coord_max = cc_verts.max()
        cc_coord_min = cc_verts.min()
        output = {
            'c_center': c_center.numpy(),
            'cc_verts': cc_verts.numpy(),
            'cc_coord_max': cc_coord_max.numpy(),
            'cc_coord_min': cc_coord_min.numpy()
        }
        return output

    def posed_smpl(self, poses, Rh_smpl, Th_smpl):
        smpl_output = self.forward(poses, Rh_smpl, Th_smpl)
        return smpl_output

    def posed_zju(self, poses, Rh_zju, Th_zju):
        R_zju = cv2.Rodrigues(Rh_zju)[0]
        Rh_smpl = np.zeros((1, 3), dtype=np.float32) 
        Th_smpl = np.zeros((1, 3), dtype=np.float32)
        smpl_output = self.forward(poses, Rh_smpl, Th_smpl)
        zju_output = {
            'zju_verts': smpl_output['smpl_verts']@R_zju.T + Th_zju,
            'zju_jnts': smpl_output['smpl_jnts']@R_zju.T + Th_zju
        }
        return zju_output

    def Th_zju2smpl(self, poses, Rh_zju, Th_zju):
        zju_output = self.posed_zju(poses, Rh_zju, Th_zju)
        smpl_output = self.posed_smpl(poses, Rh_zju, Th_zju)
        Th_smpl = Th_zju + (zju_output['zju_verts']-smpl_output['smpl_verts']).numpy().mean(axis=1)
        return Th_smpl


    def forward(self, poses, Rh, Th):
        """return SMPL output from params
        poses: [1, 69]
        Rh: [1, 3]
        Th: [1, 3]

        Returns:
            smpl_verts: vertices. shape: [B, 6893. 3]
            smpl_tfs: bone transformations. shape: [B, 24, 4, 4]
            smpl_jnts: joint positions. shape: [B, 25, 3]
        """

        Th = torch.from_numpy(Th).reshape(1, -1)
        poses = torch.from_numpy(poses).reshape(1, -1)
        Rh = torch.from_numpy(Rh).reshape(1, -1)

        smpl_output = self.smpl.forward(betas=self.betas,
                                transl=torch.zeros_like(Th),#.cuda(),
                                body_pose=poses,
                                global_orient=Rh,#.cuda(),
                                return_verts=True,
                                return_full_pose=True)

        verts = smpl_output.vertices.clone()
        joints = smpl_output.joints.clone()
        tf_mats = smpl_output.T.clone()
        Jtrs = smpl_output.J.clone()
        rots_full = smpl_output.rot_mats.clone()
        rots = rots_full.clone()
        rots[0, 0] = torch.eye(3)

        output = {}
        output['smpl_verts'] = verts + Th.unsqueeze(1) # vertices
        output['smpl_jnts'] = joints + Th.unsqueeze(1) # posed joint locations
        output['smpl_tfs'] = tf_mats # bone transformations, without translation, used in linear blend skinning
        output['smpl_Jtrs'] = Jtrs # Joint locations in T pose
        output['smpl_rots'] = rots # rotation matrices
        output['smpl_rots_full'] = rots_full
        
        
        return output
    
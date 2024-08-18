import numpy as np
from AvatarPose.utils.keypoints_utils import  keypoint_projection
from AvatarPose.utils.file_utils import read_json
import cv2
from confs.config import CONFIG

def get_keypoints3d(root):
    #{pid: keypoints3d, pid: keypoints3d}
    data = read_json(root)
    keypoints3d_dict = {}
    for i in range(len(data)):
        pid = data[i]['id']
        keypoints3d = data[i]['keypoints3d']
        name = f'human_{pid}'
        keypoints3d = np.array(keypoints3d, dtype=np.float32)
        keypoints3d_dict[name] = keypoints3d
    return keypoints3d_dict


colors_bar_rgb = [
    (28,120,255),
    (255,120,28),
    (74,  189,  172), # green
    (219, 58, 52), # red
    (100, 100, 100),
    (160, 32, 240),
    (77, 40, 49), # brown
    (255, 200, 87), # yellow
    (94, 124, 226), # 青色
    (8, 90, 97), # blue
    ( 166,  229,  204), # mint

]

colors_table = {
    'b': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [ 166/255.,  229/255.,  204/255.],
    '_mint2': [ 202/255.,  229/255.,  223/255.],
    '_green': [ 153/255.,  216/255.,  201/255.],
    '_green2': [ 171/255.,  221/255.,  164/255.],
    'r': [ 251/255.,  128/255.,  114/255.],
    '_orange': [ 253/255.,  174/255.,  97/255.],
    'y': [ 250/255.,  230/255.,  154/255.],
    'g':[0,255/255,0],
    'k':[0,0,0],
    '_r':[255/255,0,0],
    '_g':[0,255/255,0],
    '_b':[0,0,255/255],
    '_k':[0,0,0],
    '_y':[255/255,255/255,0],
    'purple':[128/255,0,128/255],
    'smap_b':[51/255,153/255,255/255],
    'smap_r':[255/255,51/255,153/255],
    'person': [255/255,255/255,255/255],
    'handl': [255/255,51/255,153/255],
    'handr': [51/255,255/255,153/255],
}

def get_rgb(index):
    if isinstance(index, int):
        if index == -1:
            return (255, 255, 255)
        if index < -1:
            return (0, 0, 0)
        # elif index == 0:
        #     return (245, 150, 150)
        col = list(colors_bar_rgb[index%len(colors_bar_rgb)])[::-1]
    elif isinstance(index, str):
        col = colors_table.get(index, (1, 0, 0))#
        col = tuple([int(c*255) for c in col[::-1]])
    else:
        raise TypeError('index should be int or str')
    return col


def plot_bbox(img, bbox, pid, scale=1, vis_id=True):
    # 画bbox: (l, t, r, b)
    x1, y1, x2, y2, c = bbox
    if c < 0.01:return img
    x1 = int(round(x1*scale))
    x2 = int(round(x2*scale))
    y1 = int(round(y1*scale))
    y2 = int(round(y2*scale))
    color = get_rgb(pid)
    lw = max(img.shape[0]//300, 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    if vis_id:
        font_scale = img.shape[0]/1000
        cv2.putText(img, '{}'.format(pid), (x1, y1+int(25*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)


def plot_keypoints(img, points, pid, config, vis_conf=False, use_limb_color=True, lw=10, fliplr=False):
    lw = max(lw, 2)
    H, W = img.shape[:2]
    for ii, (i, j) in enumerate(config['kintree']):
        if i >= len(points) or j >= len(points):
            continue
        if (i >25 or j > 25) and config['nJoints'] != 42:
            _lw = max(int(lw/4), 1)
        else:
            _lw = lw
        pt1, pt2 = points[i], points[j]
        if fliplr:
            pt1 = (W-pt1[0], pt1[1])
            pt2 = (W-pt2[0], pt2[1])
        if use_limb_color:
            col = get_rgb(config['colors'][ii])
        else:
            col = get_rgb(pid)
        if pt1[-1] > 0.01 and pt2[-1] > 0.01:
            image = cv2.line(
                img, (int(pt1[0]+0.5), int(pt1[1]+0.5)), (int(pt2[0]+0.5), int(pt2[1]+0.5)),
                col, _lw)
    for i in range(min(len(points), config['nJoints'])):
        x, y = points[i][0], points[i][1]
        if fliplr:
            x = W - x
        c = points[i][-1]
        if c > 0.01:
            text_size = img.shape[0]/1000
            col = get_rgb(pid)
            # radius = int(lw/1.5)
            radius = int(lw*1.5)
            if i > 25 and config['nJoints'] != 42:
                radius = max(int(radius/4), 1)
            cv2.circle(img, (int(x+0.5), int(y+0.5)), radius, col, -1)
            if vis_conf:
                cv2.putText(img, '{:.1f}'.format(c), (int(x), int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, text_size, col, 2)
                
def _keypoints3d_projection_1v1f1p(keypoints3d, P, pid, img, vis_id=True):
    k2d, bbox = keypoint_projection(P, keypoints3d)
    # plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
    plot_keypoints(img, k2d, pid=pid, config=CONFIG['skel19'], use_limb_color=False, lw=7)
    # plot_keypoints(img, k2d, pid=pid, config=CONFIG['shelf'], use_limb_color=False, lw=5)
    # plot_keypoints(img, k2d, pid=pid, config=CONFIG['panoptic15'], use_limb_color=False, lw=8)
    
    return img

def vis_keypoints3d(keypoints3d_dict, P, img, vis_id=True):
    for pid, val in keypoints3d_dict.items():
        img = _keypoints3d_projection_1v1f1p(val, P, pid, img, vis_id=vis_id)
    return img

def vis_kpts(predicts, P, img, vis_id=True, key = 'kpts'):
    if len(predicts) == 0:
        return img
    for name in sorted(predicts['names_all']):
        pid = int(name.split('_')[-1])
        img = _keypoints3d_projection_1v1f1p(predicts[name][key][0], P, pid, img, vis_id=vis_id)
    return img

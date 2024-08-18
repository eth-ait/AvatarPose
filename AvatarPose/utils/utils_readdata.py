import cv2
import numpy as np
from .file_utils import read_json

def img_scale(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.clip(img, 0.0, 255.0)
    img = img.astype(np.float32)/255.0
    return img

def img_scale_inv(img):
    
    img = (np.clip(img, 0, 1.)*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def read_json_with_cache(filename, file_cache):
    if filename not in file_cache.keys():
        data = read_json(filename)
        file_cache[filename] = data
    return file_cache[filename]

def scale_and_undistort(img, cam_info, camname, undistort=True, res=[512, 512], flags=cv2.INTER_LINEAR):
    K = cam_info['K']
    D = cam_info['dist']
    scale = (res[0]/img.shape[0], res[1]/img.shape[1])
    if undistort and np.linalg.norm(D)>0:
        # img_undis1 = Undistort.image(img, K, D, camname)
        img = cv2.undistort(img, K, D, None)
    img_wrap = cv2.resize(img, (res[1],  res[0]), interpolation=flags)
    img = img_scale(img_wrap)

    return img, scale
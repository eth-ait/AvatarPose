import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
from scipy.ndimage import binary_dilation
import gc
from glob import glob

def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))
def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)
def draw_mask(img, mask, alpha=0.7, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]
        # colors=[[255, 180, 28], [28, 180, 255]]
        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
                # color = colors[id]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)


# choose good parameters in sam_args based on the first frame segmentation result
# other arguments can be modified in model_args.py
# note the object number limit is 255 by default, which requires < 10GB GPU memory with amp
sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    }

# Set Text args
'''
parameter:
    grounding_caption: Text prompt to detect objects in key-frames
    box_threshold: threshold for box 
    text_threshold: threshold for label(text)
    box_size_threshold: If the size ratio between the box and the frame is larger than the box_size_threshold, the box will be ignored. This is used to filter out large boxes.
    reset_image: reset the image embeddings for SAM
'''
grounding_caption = "humans"
box_threshold, text_threshold, box_size_threshold, reset_image = 0.35, 0.5, 0.5, True
    
# For every sam_gap frames, we use SAM to find new objects and add them for tracking
# larger sam_gap is faster but may not spot new objects in time
segtracker_args = {
    'sam_gap': 20, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
}


def extract_sta_mask(root, dataname, debug=False):
    maskname = 'mask_sta'
    data_root = os.path.join(root, dataname)
    video_names = glob(os.path.join(data_root,  'images/*'))
    camnames = [os.path.basename(name) for name in video_names]
    for cam in camnames:
        io_args = {
            'input_video': os.path.join(data_root, f'videos/{cam}.mp4'),
            'output_mask_dir': os.path.join(data_root, f'{maskname}/{cam}/all'), # save pred masks
            # 'output_video': f'./hi4d/hug14/{video_name}_seg.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
            # 'output_gif': f'./hi4d/hug14/{video_name}_seg.gif', # mask visualization
        }
        campath = os.path.join(io_args['output_mask_dir'])
        os.makedirs(campath, exist_ok=True)     

        # confirm start frame
        image_names = glob(os.path.join(data_root, 'images', cam, '*'))
        frames = [int(os.path.basename(name).split('.')[0]) for name in image_names]
        start_frame = min(frames)
        # source video to segment
        cap = cv2.VideoCapture(io_args['input_video'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        # output masks
        output_dir = io_args['output_mask_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pred_list = []

        torch.cuda.empty_cache()
        gc.collect()
        sam_gap = segtracker_args['sam_gap']
        frame_idx = 0
        segtracker = SegTracker(segtracker_args, sam_args, aot_args)
        segtracker.restart_tracker()

        with torch.cuda.amp.autocast():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                if frame_idx == 0:
                    pred_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                    # pred_mask = cv2.imread('./debug/first_frame_mask.png', 0)
                    torch.cuda.empty_cache()
                    gc.collect()
                    segtracker.add_reference(frame, pred_mask)
                elif (frame_idx % sam_gap) == 0:
                    seg_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                    # save_prediction(seg_mask, './debug/seg_result', str(frame_idx)+'.png')
                    torch.cuda.empty_cache()
                    gc.collect()
                    track_mask = segtracker.track(frame)
                    # save_prediction(track_mask, './debug/aot_result', str(frame_idx)+'.png')
                    # find new objects, and update tracker with new objects
                    new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
                    if np.sum(new_obj_mask > 0) >  frame.shape[0] * frame.shape[1] * 0.4:
                        new_obj_mask = np.zeros_like(new_obj_mask)
                    # save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')
                    pred_mask = track_mask + new_obj_mask
                    # segtracker.restart_tracker()
                    segtracker.add_reference(frame, pred_mask)
                else:
                    pred_mask = segtracker.track(frame,update_memory=True)
                torch.cuda.empty_cache()
                gc.collect()
                frame_name = frame_idx+start_frame
                save_prediction(pred_mask,output_dir,f'{frame_name:06d}.png')
                
                if debug:
                    masked_frame = draw_mask(frame,pred_mask) 
                    os.makedirs(os.path.join(f'./debug/{dataname}/{cam}/masked_result'),exist_ok=True)  
                    cv2.imwrite(os.path.join(f'./debug/{dataname}/{cam}/masked_result',f'{frame_name:06d}.png'),masked_frame)
                    pred_list.append(pred_mask)
                    print(f'save debug results in ./debug/{dataname}/{cam}/masked_result')
                
                
                print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')
                frame_idx += 1
            cap.release()
            print('\nfinished')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    extract_sta_mask(args.data_root, args.seq, args.debug)
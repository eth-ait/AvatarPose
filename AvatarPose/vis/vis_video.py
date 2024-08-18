import cv2
import os
from glob import glob

def combine_video(image_root, video_root, video_name=None, fps=30):
    os.makedirs(video_root, exist_ok=True)
    images = sorted(glob(f'{image_root}/*.jpg'))
    if len(images) == 0:
        images = sorted(glob(f'{image_root}/*.png'))
    frame_width, frame_height = cv2.imread(images[0]).shape[1], cv2.imread(images[0]).shape[0]
    if video_name is None:
        video_name = os.path.basename(image_root)
    videoname = f'{video_root}/{video_name}.mp4'
    video_writer = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    for image in images:
        video_writer.write(cv2.imread(image))
    print('saved', videoname)
    
    
def combine_videos(image_root, video_root, fps=30):
    roots_1 = glob(f'{image_root}/*')
    for root_1 in roots_1:
        video_name = os.path.basename(root_1)
        combine_video(root_1, video_root, video_name, fps)
        
def combine_rgb_err_videos(image_root, video_root, start_frame, fps=30):
    roots_1 = glob(f'{image_root}/*')
    os.makedirs(video_root, exist_ok=True)
    for root_1 in roots_1:
        video_name = 'rgb_error_' + os.path.basename(root_1)
        print(f'{root_1}/rgb_err/')
        os.system(f'ffmpeg -framerate {fps} -start_number {start_frame} -i {root_1}/rgb_err/rgb_err_%d.png -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac -r {fps} {video_root}/{video_name}.mp4')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--video_root', type=str, required=True)
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    
    combine_rgb_err_videos(args.image_root, args.video_root, args.start_frame, args.fps)
    
    
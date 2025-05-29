import os
import sys
import time
import argparse
import shutil
import cv2
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="ExtractFrame")
    parser.add_argument("--data_dir", default="/your_video_path", type=str, required=False)
    parser.add_argument("--output_dir", default="/output_path", type=str, required=False)
    parser.add_argument('--fps', default=1, required=False, type=float)
    parser.add_argument('--filter_frame_front', default=0, required=False, type=int, help='Filter a few frames before saving to prevent useless information at the beginning')
    parser.add_argument('--filter_frame_last', default=0, required=False, type=int, help='Filter a few frames before saving to prevent unwanted information at the end of the video')
    return parser.parse_args()

def ExtractFrame_cv():
    args = parse_args()
    extensions = ['.mp4']
    for extension in extensions:
        video_files = glob.glob(f"{args.data_dir}/*{extension}")
        for src_file in video_files:
            try:
                cap = cv2.VideoCapture(src_file)
                FPS = cap.get(cv2.CAP_PROP_FPS)
                frameRate = int(int(FPS)/float(args.fps))
                totalFrame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                print("frameRate: ", frameRate)
                print('total video frame:',totalFrame)
                print('total video time:',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/FPS))
            except Exception as e:
                print(src_file, e)
                continue
            
            video_base_name = src_file.split("/")[-1].split(".")[0]
            output_dir =f"{args.output_dir}/{video_base_name}"
            os.makedirs(output_dir, exist_ok=True)

            frame_idx = 0
            frameAll = 0
            previous_frame_time = -1000 

            while(True):
                ret, frame = cap.read()
                if ret:
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                    interval = 1000.0 / args.fps
                    if (current_time - previous_frame_time) >= interval and int(frame_idx / frameRate) >= int(args.filter_frame_front) and frame_idx <= totalFrame - int(args.filter_frame_last) * frameRate:
                        save_path = output_dir + f"/{video_base_name}" + "_%04d"%frame_idx + ".jpg"
                        cv2.imwrite(save_path, frame)
                        previous_frame_time = current_time
                        frameAll+=1
                    frame_idx+=1
                else:
                    print("The video has been framed")
                    print("idx: ", frame_idx)
                    print("frameAll: ", frameAll)
                    break
            cap.release()

if __name__ == '__main__':
    print("Start")
    ExtractFrame_cv()
    
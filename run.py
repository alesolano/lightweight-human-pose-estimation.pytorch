import os
import argparse

import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.infer import evaluate_video


class Run():

    def __init__(self, cpu=True):
        self.cpu = cpu

        self.net = self.load_model()

        self.net = self.net.eval()
        if not self.cpu: self.net = self.net.cuda()

    def load_model(self, model_path='./models/openpose_model.pth'):
        # If model does not exist, download
        if not os.path.isfile(model_path):
            self.download_model(model_path)

        # Load model
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(model_path, map_location='cpu')
        load_state(net, checkpoint)
        return net

    
    def download_model(self, model_path):
        import urllib.request

        url = 'https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth'

        print(f'Downloading model to {model_path}...')
        urllib.request.urlretrieve(url, model_path)
        print('Download complete!')


    def get_bboxes_from_video(self, video_path, draw=False):

        bboxes_per_frame = {}

        poses_per_frame, img_shape = evaluate_video(self.net, video_path, self.cpu, draw=draw)
        
        img_height, img_width = img_shape[:2]

        for frame in poses_per_frame:
            bboxes_per_frame[frame] = []

            for pose in poses_per_frame[frame]:
                x_min = pose.bbox[0]/img_width
                y_min = pose.bbox[1]/img_height
                x_max = (pose.bbox[0] + pose.bbox[2])/img_width
                y_max = (pose.bbox[1] + pose.bbox[3])/img_height

                bboxes_per_frame[frame].append((x_min, y_min, x_max, y_max))

        return bboxes_per_frame




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='path to the video file')
    parser.add_argument('--draw', type=bool, default=False, help='flag to draw the poses and save the images')
    args = parser.parse_args()

    r = Run()
    boxes = r.get_bboxes_from_video(args.video_path, args.draw)
    print(boxes)



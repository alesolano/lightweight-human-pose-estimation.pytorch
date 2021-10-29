import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.pose import Pose
from modules.infer import infer_fast, VideoReader
from modules.keypoints import extract_keypoints, group_keypoints

# Load model
model_path = './models/openpose_model.pth'
net = PoseEstimationWithMobileNet()
checkpoint = torch.load(model_path, map_location='cpu')
load_state(net, checkpoint)

# Parameters
height_size = 256
cpu = True
#track = 1
#smooth = 1

# 
video_name = 'veriff9'
video_path = f'task_data/{video_name}.mp4'
frame_provider = VideoReader(video_path)



net = net.eval()
if not cpu:
    net = net.cuda()

stride = 8
upsample_ratio = 4
num_keypoints = Pose.num_kpts
previous_poses = []
delay = 1

Pose.reset_id()

output_imgs = []
img_i = -1
skip = 10

for img in frame_provider:
    
    img_i += 1
    if img_i % skip != 0: continue
    
    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    # if track:
    #     track_poses(previous_poses, current_poses, smooth=smooth)
    #     previous_poses = current_poses
    
    for pose in current_poses:
        pose.draw(img)
    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    for pose in current_poses:
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        # if track:
        #     cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
        #                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    
    # Write to image
    img_path = f'./{img_i:05}.png'
    print(img_path)
    for pose in current_poses:
        print(pose.bbox, img.shape)
    cv2.imwrite(img_path, img)
        


import torch
from detect_main import detect_image_no_cmd
from demo_fastflow import optical_flow_estimation
import implementation as model

import os
import helper
# print('Starting...')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.is_available():
#     print(f"{torch.cuda.device_count()} devices available")
# optical_flow_estimation('videos/Stealing108_x264.mp4', 'weights/fastflownet_gtav.pth', testing=False, save_in_rgb=False, save_img=True)
# # detect_image_no_cmd('videos/Burglary001_x264.mp4', 'weights/yolov7.pt', True, 640, torch.device('cuda'))

# pipeline = model.FastFlowYOLOPipeline(
# 'weights/yolov7.pt', 'weights/fastflownet_gtav.pth', 'inference', False, 'test', binary_tresh=200)

# optical_flow_estimation('videos/Stealing108_x264.mp4', 'weights/fastflownet_gtav.pth',
# testing=False, save_in_rgb=False, save_img=True)
# pipeline.detect_and_optical_flow(
# 'videos/Robbery098_x264.mp4')
# helper.apply_to_folders(
# pipeline, ['resized/*.mp4', 'videos/Stealing*.mp4', 'videos/Robbery*.mp4', 'videos/Burglary*.mp4'], 'Opening')
helper.test_data("7th_change/", "analysis/7th_change_test/")

print('Done.')

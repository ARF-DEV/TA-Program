import torch
from detect_main import detect_image_no_cmd

print('Starting...')
detect_image_no_cmd('images/horses.jpg', 'weights/yolov7.pt', True, 640, 'output', torch.device('cpu'))
print('Done.')
import torch
from detect_main import detect_image_no_cmd

print('Starting...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"{torch.cuda.device_count()} devices available")
    print('Using GPU')
else:
    print('Using CPU')
detect_image_no_cmd('videos/Stealing/Stealing002_x264.mp4', 'weights/yolov7.pt', True, 640, 'output', torch.device('cuda'))
print('Done.')
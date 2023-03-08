import torch
from detect_main import detect_image_no_cmd
from demo_fastflow import optical_flow_estimation

print('Starting...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"{torch.cuda.device_count()} devices available")
    print('Using GPU')
else:
    print('Using CPU')
optical_flow_estimation('videos/Stealing002_x264.mp4', 'weights/fastflownet_ft_mix.pth', False, False, True)
# detect_image_no_cmd('videos/Burglary001_x264.mp4', 'weights/yolov7.pt', True, 640, torch.device('cuda'))
print('Done.')
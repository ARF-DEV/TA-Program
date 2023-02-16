import torch
from detect_main import detect_image_no_cmd

print('Starting...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"{torch.cuda.device_count()} devices available")
    print('Using GPU')
else:
    print('Using CPU')
detect_image_no_cmd('images/horses.jpg', 'weights/yolov7.pt', True, 640, 'output', torch.device('cuda'))
print('Done.')
import implementation as model
import helper
print('Starting...')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.is_available():
#     print(f"{torch.cuda.device_count()} devices available")
# optical_flow_estimation('videos/Stealing108_x264.mp4', 'weights/fastflownet_gtav.pth', testing=False, save_in_rgb=False, save_img=True)
# # detect_image_no_cmd('videos/Burglary001_x264.mp4', 'weights/yolov7.pt', True, 640, torch.device('cuda'))

pipeline = model.FastFlowYOLOPipeline(
    'weights/yolov7.pt', 'weights/fastflownet_gtav.pth', 'inference',
    False, binary_tresh=200, binary_sum_tresh=400, is_opening=False)

# optical_flow_estimation('videos/Stealing108_x264.mp4', 'weights/fastflownet_gtav.pth',
# testing=False, save_in_rgb=False, save_img=True)
# pipeline.detect_and_optical_flow(
# 'videos/Robbery098_x264.mp4')
helper.apply_to_folders(
    pipeline, ['resized/*.mp4', 'videos/Stealing*.mp4', 'videos/Robbery*.mp4', 'videos/Burglary*.mp4'])


print('Done.')

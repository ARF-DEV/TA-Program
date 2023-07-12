import argparse
import implementation as model
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str,
                    help='Path to video file')
parser.add_argument('-o', '--output', type=str, help='Path to output file')

args = parser.parse_args()

video_path = Path(args.video)
if video_path.exists() and video_path.is_file():
    if str(video_path).split('.')[1] not in ['mp4', 'avi']:
        print('Video file must be mp4 or avi')
        exit()
    print('Video file found. Processing...')
    pipeline = model.FastFlowYOLOPipeline(
        'weights/yolov7.pt', 'weights/fastflownet_gtav.pth', 'inference/__ini_yg_paling_baru_confidence_score/300_bst_new/',
        False, binary_tresh=200, binary_sum_tresh=300, is_opening=False)
    pipeline.detect_and_optical_flow(
        args.video, debug=False, output_path=args.output)
else:
    print('Video file not found.')
    exit()

print('Done.')

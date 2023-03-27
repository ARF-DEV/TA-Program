import glob
from implementation import FastFlowYOLOPipeline, resize_video


def apply_to_folders(model, paths):
    if not isinstance(model, FastFlowYOLOPipeline):
        raise Exception('model must be an instance of FastFlowYOLOPipeline')
    for path in paths:
        files = glob.glob(path)
        print(files)
        for file in files:
            print(f'processing {path}')
            model.detect_and_optical_flow(file)
    

def resize_videos(pathname):
    paths = glob.glob(pathname)
    for path in paths:
        print(f'resizing {path}')
        saved_path = resize_video(path, (320, 240)) 
    print(f'save in {saved_path}')
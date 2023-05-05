import glob
from implementation import FastFlowYOLOPipeline, resize_video
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def apply_to_folders(model, paths, mode='normal'):
    if not isinstance(model, FastFlowYOLOPipeline):
        raise Exception('model must be an instance of FastFlowYOLOPipeline')
    all_files = []
    for path in paths:
        files = glob.glob(path)
        all_files = all_files + files

    mean_median_all = pd.DataFrame(columns=['video', 'mean', 'median'])
    print(f'processing {len(all_files)} files')
    for idx, file in enumerate(all_files):
        print(f'{idx+1}. {file}')
    for file in all_files:
        print(f'processing {file}')
        save_dir, mean, median = model.detect_and_optical_flow(file, mode=mode)
        mean_median_df = pd.DataFrame({
            'video': [file],
            'mean': [mean],
            'median': [median]
        })
        mean_median_all = pd.concat(
            [mean_median_all, mean_median_df], ignore_index=True)

    mean_median_all.to_csv(Path(save_dir / "mean_median_all.csv"), index=False)


def resize_videos(pathname):
    paths = glob.glob(pathname)
    for path in paths:
        print(f'resizing {path}')
        saved_path = resize_video(path, (320, 240))
    print(f'save in {saved_path}')


def test_data(prediction_folder):
    labels = glob.glob('labels/*')
    print(len(labels))
    for l in labels:
        video_name = ".".join(l.split(" ")[-1].split(".")[0:2])
        path = Path("inference") / prediction_folder / \
            video_name / "fastflow_movement.csv"
        if path.exists():
            # print(path)
            pred = pd.read_csv(path)
            actual = pd.read_csv(l)
            print(f"pred path: {path}")
            print(f"actual path: {l}")
            print()
            min_length = min(actual['Penting'].size, pred['Penting'].size)
            c = pd.crosstab(actual['Penting'].iloc[0: min_length], pred['Penting'].iloc[0: min_length], rownames=[
                            'actual'], colnames=['pred'])
            print(c)
            print(
                f"precision: {precision_score(actual['Penting'].iloc[0: min_length], pred['Penting'].iloc[0: min_length])}")
            print(
                f"recall: {recall_score(actual['Penting'].iloc[0: min_length], pred['Penting'].iloc[0: min_length])}")
            print(
                f"f1: {f1_score(actual['Penting'].iloc[0: min_length], pred['Penting'].iloc[0: min_length])}")
            # pred['Penting'] = pred['Penting'].astype(int)
            # testa['Penting'] = testa['Penting'].astype(int)
        # else:
            # print(path)

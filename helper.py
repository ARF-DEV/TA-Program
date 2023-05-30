import glob
from implementation import FastFlowYOLOPipeline, resize_video
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sn


def apply_to_folders(model, paths):
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
        save_dir, mean, median = model.detect_and_optical_flow(
            file, debug=True)
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


def test_data(prediction_folder, dst_folder):
    if not Path(dst_folder).exists():
        Path(dst_folder).mkdir()
    labels = glob.glob('labels/*')
    all_metrics_dst = Path(dst_folder) / "metrics.csv"
    desc_metrics_dst = Path(dst_folder) / "mean_metrics.csv"
    print(len(labels))
    df_all = pd.DataFrame(columns=["video", "precision", "recall", "f1"])
    n = 0
    for l in labels:
        video_name = ".".join(l.split(" ")[-1].split(".")[0:2])
        path = Path("inference") / prediction_folder / \
            video_name / "fastflow_movement.csv"
        mean_ps = 0
        mean_rs = 0
        mean_fs = 0
        if path.exists():
            image_dir = Path(dst_folder) / (video_name+".png")
            pred = pd.read_csv(path)
            actual = pd.read_csv(l)
            print(f"pred path: {path}")
            print(f"actual path: {l}")
            print()
            if actual['Penting'].size != pred['Penting'].size:
                vid_name = video_name + "_" + \
                    str(100 * round(abs(actual['Penting'].size -
                        pred['Penting'].size) / actual['Penting'].size, 3))
            else:
                vid_name = video_name

            min_length = min(actual['Penting'].size, pred['Penting'].size)
            c = confusion_matrix(
                actual['Penting'].iloc[0: min_length], pred['Penting'].iloc[0: min_length])
            ps = precision_score(
                actual['Penting'].iloc[0: min_length], pred['Penting'].iloc[0: min_length])
            rs = recall_score(
                actual['Penting'].iloc[0: min_length], pred['Penting'].iloc[0: min_length])
            fs = f1_score(actual['Penting'].iloc[0: min_length],
                          pred['Penting'].iloc[0: min_length])
            mean_ps += ps
            mean_rs += rs
            mean_fs += fs
            df_all = pd.concat([df_all, pd.DataFrame({
                "video": [vid_name],
                "precision": [ps],
                "recall": [rs],
                "f1": [fs]
            })], ignore_index=True)
            # df_all = df_all.concat({
            #     "precision": ps,
            #     "recall": rs,
            #     "f1": fs
            # }, ignore_index=True)
            n += 1
            print(c)
            print(
                f"precision: {ps}")
            print(
                f"recall: {rs}")
            print(
                f"f1: {fs}")
            fig = sn.heatmap(c, annot=True, fmt='d').get_figure()
            plt.title('Confusion Matrix : ' + video_name)
            plt.ylabel('True Label')
            plt.xlabel('Predicated Label')
            plt.savefig(image_dir)
            plt.close(fig)

    # df_mean = df_mean.concat({
    #     "precision": mean_ps/n,
    #     "recall": mean_rs/n,
    #     "f1": mean_fs/n
    # }, ignore_index=True)

    df_all.to_csv(all_metrics_dst, index=False)
    df_all.describe().to_csv(desc_metrics_dst)

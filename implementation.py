import torch
from models.FastFlowNet_ import FastFlowNet
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized, select_device
from utils.general import non_max_suppression, scale_coords, set_logging, increment_path
from utils.datasets import LoadImages
import cv2
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from flow_vis import flow_to_color
from demo_fastflow import centralize
import pandas as pd
import os
import random


class FastFlowYOLOPipeline:
    def __init__(self, yolo_weights, fastflownet_weights, inference_path, path_exist_ok, folder_name, binary_tresh):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.FFN = FastFlowNet().cuda().eval()
        self.FFN.load_state_dict(torch.load(fastflownet_weights))
        self.YOLO = attempt_load(yolo_weights, map_location=self.device).eval()
        self.image_size = 320
        self.stride = int(self.YOLO.stride.max())  # model stride
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.yolo_names = self.YOLO.module.names if hasattr(
            self.YOLO, 'module') else self.YOLO.names
        self.class_colors = [
            [random.randint(0, 255) for _ in range(3)] for _ in self.yolo_names]
        self.div_flow = 20.0
        self.div_size = 64
        self.inference_path = inference_path
        self.path_exist_ok = path_exist_ok
        self.folder_name = folder_name
        select_device(self.device.type)
        if self.device != 'cpu':
            print("USING GPU")
            self.half = True
            self.YOLO.half()  # to FP16
        self.binary_tresh = binary_tresh

    def detect(self, img, im0):
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.YOLO(img, augment=False)[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        t3 = time_synchronized()
        print(f'Detection time: {t2 - t1:.3f}s')
        print(f'NMS time: {t3 - t2:.3f}s')

        detected_classes_name = []
        for i, det in enumerate(pred):
            s = ''
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                detected_classes_name.append(self.yolo_names[int(c)])
                # add to string
                s += f"{n} {self.yolo_names[int(c)]}{'s' * (n > 1)}, "
            print(f"class: {s}")
        return det, detected_classes_name

    def optical_flow(self, img1, img2):
        print(f"\n {img1.shape}")
        # img1 = cv2.medianBlur(img1, 3)
        # img2 = cv2.medianBlur(img2, 3)
        img1 = torch.from_numpy(img1).float().permute(
            2, 0, 1).unsqueeze(0)/255.0
        img2 = torch.from_numpy(img2).float().permute(
            2, 0, 1).unsqueeze(0)/255.0
        # img1 = cv2.medianBlur(img1.squeeze(0).permute(1, 2, 0).numpy(), 3)
        # img2 = cv2.medianBlur(img2.squeeze(0).permute(1, 2, 0).numpy(), 3)
        img1, img2, _ = centralize(img1, img2)
        height, width = img1.shape[-2:]
        orig_size = (int(height), int(width))
        if height % self.div_size != 0 or width % self.div_size != 0:
            input_size = (
                int(self.div_size * np.ceil(height / self.div_size)),
                int(self.div_size * np.ceil(width / self.div_size))
            )
            img1 = F.interpolate(img1, size=input_size,
                                 mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=input_size,
                                 mode='bilinear', align_corners=False)
        else:
            input_size = orig_size
        input_t = torch.cat([img1, img2], 1).cuda()
        with torch.no_grad():
            output = self.FFN(input_t).data
        flow = self.div_flow * \
            F.interpolate(output, size=input_size,
                          mode='bilinear', align_corners=False)
        if input_size != orig_size:
            scale_h = orig_size[0] / input_size[0]
            scale_w = orig_size[1] / input_size[1]
            flow = F.interpolate(flow, size=orig_size,
                                 mode='bilinear', align_corners=False)
            flow[:, 0, :, :] *= scale_w
            flow[:, 1, :, :] *= scale_h
        flow = flow[0].cpu().permute(1, 2, 0).numpy()
        flow_color = flow_to_color(flow, convert_to_bgr=True)
        flow_gray = cv2.cvtColor(flow_color, cv2.COLOR_BGR2GRAY)
        im_bw = cv2.threshold(flow_gray, self.binary_tresh,
                              255, cv2.THRESH_BINARY_INV)[1]
        return flow_color, flow_gray, im_bw

    def detect_and_optical_flow(self, source, mode='normal'):
        set_logging()
        image_size = check_img_size(
            self.image_size, s=self.stride)  # check img_size
        dataset = LoadImages(source, img_size=image_size, stride=self.stride)
        save_dir = Path(increment_path(Path(self.inference_path) /
                        Path(dataset.get_file_name()).name, exist_ok=self.path_exist_ok))  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)
        t1 = time_synchronized()
        frame = 1
        p_path, p_img, p_im0, p_vid_cap = next(dataset)
        frame_flow_sum_data = pd.DataFrame(columns=['frame', 'flow_sum'])
        frame_flow_movement = pd.DataFrame(
            columns=['Frame', 'Penting'])
        frame_flow_gray_sum_data = pd.DataFrame(columns=['frame', 'flow_sum'])
        frame_flow_binary_sum_data = pd.DataFrame(
            columns=['frame', 'flow_sum'])
        frame_important = pd.DataFrame(columns=['frame', 'objects'])

        save_path_flow = str(save_dir / "rgb_flow_video.mp4")
        save_path_flow_gray = str(save_dir / "gray_flow_video.mp4")
        save_path_flow_binary = str(save_dir / "binary_flow_video.mp4")
        save_path_normal = str(save_dir / "normal_video.mp4")
        save_path_yolo = str(save_dir / "yolo_video.mp4")
        save_path_final = str(save_dir / "final_video.mp4")

        print(f"save_path: {save_dir}")
        fps = p_vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(p_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(p_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer_flow = cv2.VideoWriter(
            str(save_path_flow), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer_flow_gray = cv2.VideoWriter(
            str(save_path_flow_gray), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), 0)
        vid_writer_flow_binary = cv2.VideoWriter(
            str(save_path_flow_binary), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), 0)
        vid_writer_flow_normal = cv2.VideoWriter(
            str(save_path_normal), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer_yolo = cv2.VideoWriter(
            str(save_path_yolo), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer_final = cv2.VideoWriter(
            str(save_path_final), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for c_path, c_img, c_im0, c_vid_cap in dataset:
            frame += 1

            vid_writer_flow_normal.write(c_im0)
            # FASTFLOWNET
            t3 = time_synchronized()
            flow, flow_gray, flow_binary = self.optical_flow(p_im0, c_im0)

            # Using morphologyEx() method for opening operation (erode & dilate)
            if mode == 'Opening':
                print("Opening Method")
                kernel = np.ones((5, 5), np.uint8)
                flow_binary = cv2.morphologyEx(
                    flow_binary, cv2.MORPH_OPEN, kernel)

            flow_sum = np.sum(flow // 255.0)
            flow_gray_sum = np.sum(flow_gray // 255.0)
            flow_binary_sum = np.sum(flow_binary // 255.0)
            print(flow_binary)
            print(flow_binary.shape)
            t4 = time_synchronized()
            print(f"total sum: {flow_sum}")
            print(f"Optical flow time: {t4 - t3:.3f}s")

            # save flow_sum to a file
            new_data = pd.DataFrame({
                'frame': [frame],
                'flow_sum': [flow_sum]
            })
            new_data_gray = pd.DataFrame({
                'frame': [frame],
                'flow_sum': [flow_gray_sum]
            })
            new_data_binary = pd.DataFrame({
                'frame': [frame],
                'flow_sum': [flow_binary_sum]
            })

            frame_flow_sum_data = pd.concat(
                [frame_flow_sum_data, new_data], ignore_index=True)
            frame_flow_gray_sum_data = pd.concat(
                [frame_flow_gray_sum_data, new_data_gray], ignore_index=True)
            frame_flow_binary_sum_data = pd.concat(
                [frame_flow_binary_sum_data, new_data_binary], ignore_index=True)

            if flow_binary_sum <= 100:
                print("No motion detected")
                p_path, p_img, p_im0, p_vid_cap = c_path, c_img, c_im0, c_vid_cap
                vid_writer_flow.write(flow)
                vid_writer_flow_gray.write(flow_gray)
                vid_writer_flow_binary.write(flow_binary)
                vid_writer_flow.write(flow)
                new_movement_data = pd.DataFrame({
                    'Frame': [frame],
                    'Penting': False,
                })
                frame_flow_movement = pd.concat(
                    [frame_flow_movement, new_movement_data], ignore_index=True)
                continue

            if type(c_img) is np.ndarray:
                c_img = torch.from_numpy(c_img).to(self.device)
            else:
                c_img = c_img.to(self.device)
            c_img = c_img.half() if self.half else c_img.float()  # uint8 to fp16/32
            c_img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if type(p_img) is np.ndarray:
                p_img = torch.from_numpy(p_img).to(self.device)
            else:
                p_img = p_img.to(self.device)
            p_img = p_img.half() if self.half else p_img.float()  # uint8 to fp16/32
            p_img /= 255.0
            # YOLO
            t5 = time_synchronized()
            if c_img.ndimension() == 3:
                c_img = c_img.unsqueeze(0)
            det, detected_classes_name = self.detect(c_img, c_im0)

            im0_copy = c_im0.copy()
            if len(det):
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.yolo_names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0_copy, label=label,
                                 color=self.class_colors[int(cls)], line_thickness=1)
            t6 = time_synchronized()
            print(f"YOLO time: {t6 - t5:.3f}s")

            new_movement_data = pd.DataFrame({
                'Frame': [frame],
                'Penting': True,
            })
            frame_flow_movement = pd.concat(
                [frame_flow_movement, new_movement_data], ignore_index=True)

            if 'cars' in detected_classes_name or 'person' in detected_classes_name:
                # save the frame
                vid_writer_final.write(c_im0)
                # save frame to dataframe
                new_final = pd.DataFrame({
                    'frame': [frame],
                    'objects': [", ".join(detected_classes_name)]
                })
                frame_important = pd.concat(
                    [frame_important, new_final], ignore_index=True)

            vid_writer_flow.write(flow)
            vid_writer_flow_gray.write(flow_gray)
            vid_writer_flow_binary.write(flow_binary)

            vid_writer_yolo.write(im0_copy)

            p_path, p_img, p_im0, p_vid_cap = c_path, c_img, c_im0, c_vid_cap
            print(f"total time for frame: {((t6 - t3) / 1000):.3f}s")

        t2 = time_synchronized()
        print(f"Total time: {((t2 - t1) / 1000):.3f}s")
        frame_flow_sum_data.to_csv(
            Path(save_dir / "fastflow_rgb_sum.csv"), index=False)

        frame_flow_movement.to_csv(
            Path(save_dir / "fastflow_movement.csv"), index=False)
        frame_flow_gray_sum_data.to_csv(
            Path(save_dir / "fastflow_gray_sum.csv"), index=False)
        frame_flow_binary_sum_data.to_csv(
            Path(save_dir / "fastflow_binary_sum.csv"), index=False)
        frame_important.to_csv(
            Path(save_dir / "important_frames.csv"), index=False)
        vid_writer_flow.release()
        vid_writer_flow_gray.release()
        vid_writer_flow_binary.release()
        vid_writer_flow_normal.release()
        vid_writer_yolo.release()
        vid_writer_final.release()
        return Path(self.inference_path), frame_flow_binary_sum_data['flow_sum'].mean(), frame_flow_binary_sum_data['flow_sum'].median()


def resize_video(source, target_size):
    dataset = LoadImages(source)
    save_dir = Path(increment_path(Path("resized")))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)
    vid_writer, save_path = None, None
    for path, img, im0s, vid_cap in dataset:
        resized_frame = cv2.resize(im0s, target_size)
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        path_arr = os.path.splitext(path)
        main_path = path_arr[0] + "_resized" + path_arr[1]
        # print(f"path: {Path(main_path).name}")
        save_path = Path(save_dir / Path(main_path).name)
        if vid_writer is None:
            vid_writer = cv2.VideoWriter(
                str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, target_size)
        vid_writer.write(resized_frame)
    return save_path

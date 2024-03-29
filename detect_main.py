import argparse
import time
from pathlib import Path

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def detect_image_no_cmd(source, weights, save_txt, image_size, device):
    dict = {
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'image_size': 640,
        'classes': None,
        'agnostic_nms': False,
        'save_dir': 'inference',
        'name': 'exp',
        'exist_ok': False,
        'save_conf': False,
    }
    augment = False
    save_img = not source.endswith('.txt')  # save inference images
    vid_path, vid_writer = None, None

    save_dir = Path(increment_path(Path(dict['save_dir']) / dict['name'], exist_ok=dict['exist_ok']))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    set_logging()
    select_device(device.type)
    half = device.type != 'cpu'  # half precision only supported on CUDA    

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    image_size = check_img_size(image_size, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=image_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, image_size, image_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = image_size
    old_img_b = 1

    t0 = time.time()
    for path, img, im0, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, dict['conf_thres'], dict['iou_thres'], classes=dict['classes'], agnostic=dict['agnostic_nms'])
        t3 = time_synchronized()
        s = ''
        # print(im0s.shape)
        # print(img.shape)
        for i, det in enumerate(pred):
            p = Path(path)
            save_path = str(save_dir / p.name)  # img.jpg

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                print(f"class: {s}")
                # print(f"det shape: {det[0].shape}")
            for *xyxy, conf, cls in reversed(det): # det itu list of detections
                print(f"xyxy: {xyxy}")
                print(f"conf: {conf}")
                print(f"cls: {names[int(cls)]}")
                if save_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0) 

        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        
            # Process detections
        # for i, det in enumerate(pred):  # detections per image
        #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        #     p = Path(p)  # to Path
        #     save_path = str(save_dir / p.name)  # img.jpg
        #     txt0_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        #         # Print results
        #         for c in det[:, -1].unique():
        #             n = (det[:, -1] == c).sum()  # detections per class
        #             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             if save_txt:  # Write to file
        #                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #                 line = (cls, *xywh, conf) if dict['save_conf'] else (cls, *xywh)  # label format
        #                 with open(txt_path + '.txt', 'a') as f:
        #                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

        #             if save_img:  # Add bbox to image
        #                 label = f'{names[int(cls)]} {conf:.2f}'
        #                 plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        #     # Print time (inference + NMS)
        #     print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        #     # Save results (image with detections)
        #     if save_img:
        #         if dataset.mode == 'image':
        #             cv2.imwrite(save_path, im0)
        #             print(f" The image with the result is saved in: {save_path}")
        #         else:  # 'video' or 'stream'
        #             if vid_path != save_path:  # new video
        #                 vid_path = save_path
        #                 if isinstance(vid_writer, cv2.VideoWriter):
        #                     vid_writer.release()  # release previous video writer
        #                 if vid_cap:  # video
        #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #                 else:  # stream
        #                     fps, w, h = 30, im0.shape[1], im0.shape[0]
        #                     save_path += '.mp4'
        #                 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #             vid_writer.write(im0)

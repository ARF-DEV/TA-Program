import numpy as np
import cv2
import torch
import torch.nn.functional as F
from models.FastFlowNet_ import FastFlowNet
from utils.datasets import LoadImages
from flow_vis import flow_to_color
from pathlib import Path
from utils.general import increment_path
from utils.torch_utils import time_synchronized

div_flow = 20.0
div_size = 64


def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(
        b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean


def get_optical_flow_estimation(model, img1, img2):
    print(f"\n {img1.shape}")
    img1 = cv2.medianBlur(img1, 3)
    img2 = cv2.medianBlur(img2, 3)
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0)/255.0
    # img1 = cv2.medianBlur(img1.squeeze(0).permute(1, 2, 0).numpy(), 3)
    # img2 = cv2.medianBlur(img2.squeeze(0).permute(1, 2, 0).numpy(), 3)
    img1, img2, _ = centralize(img1, img2)

    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))
    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)),
            int(div_size * np.ceil(width / div_size))
        )
        img1 = F.interpolate(img1, size=input_size,
                             mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size,
                             mode='bilinear', align_corners=False)
    else:
        input_size = orig_size
    input_t = torch.cat([img1, img2], 1).cuda()
    output = model(input_t).data
    flow = div_flow * \
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
    return flow_to_color(flow, convert_to_bgr=True)


def optical_flow_estimation(source, weights, testing=False, save_in_rgb=True, save_img=True):
    dict = {
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'image_size': 640,
        'classes': None,
        'agnostic_nms': False,
        'save_dir': 'inference',
        'name': 'optic_flow',
        'exist_ok': False,
        'save_conf': False,
    }
    tstart = time_synchronized()
    vid_path, vid_writer = None, None

    save_dir = Path(increment_path(
        Path(dict['save_dir']) / dict['name'], exist_ok=dict['exist_ok']))
    save_dir.mkdir(parents=True, exist_ok=True)  # increment run
    model = FastFlowNet().cuda().eval()
    model.load_state_dict(torch.load(weights))

    dataset = LoadImages(source)
    if testing:
        count = 0
    c_path, c_img, c_im0, c_vid_cap = next(dataset)
    p_path, p_img, p_im0, p_vid_cap = c_path, c_img, c_im0, c_vid_cap
    for c_path, c_img, c_im0, c_vid_cap in dataset:
        if testing:
            count += 1
            if count == 10:
                break

        if c_path is None:
            vid_writer.release()
            break

        p = Path(c_path)
        save_path = str(save_dir / p.name)
        t1 = time_synchronized()
        flow_color = get_optical_flow_estimation(model, p_im0, c_im0)
        # flow_color[:, :, 0] = flow_color[:, :, 0] * 0.5 + 0.5 * p_im0[:, :, 0]
        t2 = time_synchronized()
        print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference')

        if save_img:  # Add bbox to image
            if dataset.mode == 'image':
                cv2.imwrite(save_path, flow_color)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if c_vid_cap:  # video
                        fps = c_vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(c_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(c_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, c_im0.shape[1], c_im0.shape[0]
                        save_path += '.mp4'

                    if save_in_rgb:
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    else:
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), 0)
                else:
                    if save_in_rgb:
                        vid_writer.write(flow_color)
                    else:
                        flow_gray = cv2.cvtColor(
                            flow_color, cv2.COLOR_BGR2GRAY)
                        im_bw = cv2.threshold(
                            flow_gray, 150, 255, cv2.THRESH_BINARY)[1]
                        print(im_bw.shape)
                        vid_writer.write(im_bw)
        p_path, p_img, p_im0, p_vid_cap = c_path, c_img, c_im0, c_vid_cap
    tend = time_synchronized()
    print(f'Done. ({(1E3 * (tend - tstart)):.1f}ms) Inference')

import torch
from models import FastFlowNet_ as FastFlowNet
from models.experimental import attempt_load

class FastFlowYOLOPipeline:
    def __init__(self, yolo_weights, fastflownet_weights):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FFN = FastFlowNet().cuda().eval()
        self.FFN.load_state_dict(torch.load(fastflownet_weights))
        self.YOLO = attempt_load(yolo_weights, map_location=self.device).eval()

    def detect(self, source, save_img=False, save_in_rgb=True):
        pass

    def optical_flow(self, source, save_img=False, save_in_rgb=True):
        pass

    def detect_and_optical_flow(self, source, save_img=False, save_in_rgb=True):
        pass


    
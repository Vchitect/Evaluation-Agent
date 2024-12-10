import io
import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from vbench.utils import load_video, load_dimension_info, dino_transform, dino_transform_Image
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def subject_consistency(model, video_pairs, device):
    sim = 0.0
    cnt = 0
    video_results = []

    image_transform = dino_transform(224)
    
    for info in tqdm(video_pairs):
        query = info['prompt']
        video_path = info['content_path']
        
        video_sim = 0.0

        images = load_video(video_path)
        images = image_transform(images)
            
        for i in range(len(images)):
            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image = image.to(device)
                image_features = model(image)
                image_features = F.normalize(image_features, dim=-1, p=2)
                if i == 0:
                    first_image_features = image_features
                else:
                    sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                    cur_sim = (sim_pre + sim_fir) / 2
                    video_sim += cur_sim
                    cnt += 1
            former_image_features = image_features
        sim_per_images = video_sim / (len(images) - 1)
        sim += video_sim
        video_results.append({'prompt':query, 'video_path': video_path, 'video_results': sim_per_images})


    sim_per_frame = sim / cnt
    
    return {
        "score":[sim_per_frame, video_results] 
    }


def compute_subject_consistency(video_pairs):
    device = torch.device("cuda")
    submodules_list = {
        'repo_or_dir':'facebookresearch/dino:main',
        'source':'github',
        'model': 'dino_vitb16',
    }
    dino_model = torch.hub.load(**submodules_list).to(device)

    logger.info("Initialize DINO success")

    results = subject_consistency(dino_model, video_pairs, device)
    return results

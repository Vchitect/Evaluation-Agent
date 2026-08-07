import os
import json
import numpy as np
import clip
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from vbench.utils import load_video, load_dimension_info, CACHE_DIR
from vbench.third_party.umt.datasets.video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter, 
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
)
from vbench.third_party.umt.datasets.volume_transforms import ClipToTensor
from timm.models import create_model
from vbench.third_party.umt.models.modeling_finetune import vit_large_patch16_224
from tqdm import tqdm

def build_dict():
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    path = f'{CUR_DIR}/third_party/umt/kinetics_400_categories.txt'
    results = {}
    with open(path, 'r') as f:
        cat_list = f.readlines()
        cat_list = [c.strip() for c in cat_list]
        for line in cat_list:
            cat, number = line.split('\t')
            results[number] = cat.lower()
    return results


def human_action(umt_path, video_pairs, device):
    state_dict = torch.load(umt_path, map_location='cpu')
    model = create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=400,
        all_frames=16,
        tubelet_size=1,
        use_learnable_pos_emb=False,
        fc_drop_rate=0.,
        drop_rate=0.,
        drop_path_rate=0.2,
        attn_drop_rate=0.,
        drop_block_rate=None,
        use_checkpoint=False,
        checkpoint_num=16,
        use_mean_pooling=True,
        init_scale=0.001,
    )
    data_transform = Compose([
        Resize(256, interpolation='bilinear'),
        CenterCrop(size=(224, 224)),
        ClipToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = model.to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    cat_dict = build_dict()
    cnt= 0
    cor_num = 0
    
    video_results = []
    
    
    for info in tqdm(video_pairs):
        
        query = info['prompt']
        video_path = info['content_path']
        
        # The prompt is the ground-truth Kinetics action. Deriving it from the
        # generated filename breaks as soon as filenames are shortened or gain
        # collision-resistant hashes.
        video_label_ls = " ".join(query.casefold().split())
        for prefix in ("a person is ", "person is "):
            if video_label_ls.startswith(prefix):
                video_label_ls = video_label_ls[len(prefix):]
                break
        cnt += 1
        images = load_video(video_path, data_transform, num_frames=16)
        images = images.unsqueeze(0)
        images = images.to(device)
        with torch.no_grad():
            logits = torch.sigmoid(model(images))
            results, indices = torch.topk(logits, 5, dim=1)
        indices = indices.squeeze().tolist()
        results = results.squeeze().tolist()
        results = [round(f, 4) for f in results]
        cat_ls = []
        for i in range(5):
            if results[i] >= 0.85:
                cat_ls.append(cat_dict[str(indices[i])])
        flag = False
        for cat in cat_ls:
            if cat == video_label_ls:
                cor_num += 1
                flag = True
                # print(f"{cnt}: {video_path} correct, top-5: {cat_ls}, logits: {results}", flush=True)
                break
        if flag is False:
            # print(f"{cnt}: {video_path} false, gt: {video_label_ls}, top-5: {cat_ls}, logits: {results}", flush=True)
            pass
        video_results.append({'prompt':query, 'video_path': video_path, 'video_results': flag})
    # print(f"cor num: {cor_num}, total: {cnt}")
    acc = cor_num / cnt

    return {
        "score":[acc, video_results] 
    }


def compute_human_action(video_pairs):
    device = torch.device("cuda")
    umt_path = f'{CACHE_DIR}/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth'

    results = human_action(umt_path, video_pairs, device)
    return results

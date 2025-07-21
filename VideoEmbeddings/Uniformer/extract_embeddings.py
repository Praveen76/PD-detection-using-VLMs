from uniformer import uniformer_small
from huggingface_hub import hf_hub_download
import torch
import pickle
import time

import os
from tqdm import tqdm

import numpy as np
import subprocess as sp
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader
from decord import cpu
from transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

from huggingface_hub import hf_hub_download
from app import load_video
import pandas as pd

import gc

import multiprocessing

'''
Find the GPU that has max free space
'''
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def get_embedding(video_path):
    np.random.seed(0)
    '''
    set-up device (for gpu support)
    '''
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = uniformer_small()
    # load state
    model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_small_k400_16x8.pth")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    # set to eval mode
    model = model.to(device)
    model = model.eval()

    vid = load_video(video_path).to(device)
    mean_pooled = model.get_embedding(vid).reshape(-1)
    return mean_pooled.detach()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    smile_files = pd.read_csv("/localdisk1/PARK/park_multitask_fusion/data/facial_expression_smile/facial_dataset.csv")
    finger_files = pd.read_csv("/localdisk1/PARK/park_multitask_fusion/data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
    
    smile_files = list(smile_files["Filename"])
    finger_files = list(finger_files["filename"])

    dataset = []
    count = 0

    for filename in tqdm(os.listdir("/localdisk2/park_videos/all_tasks/raw_videos/")):
        if (filename in smile_files) or (filename in finger_files):
            item = {"filename":filename}
            file_path = os.path.join("/localdisk2/park_videos/all_tasks/raw_videos/", filename)

            try:
                with multiprocessing.Pool() as pool:
                    result = pool.apply_async(get_embedding, (file_path,))
                    item["mean_pooled_embedding"] = result.get()
                    dataset.append(item)
                    count +=1
            except Exception as e:
                print(e)
                
            if count%5==0:
                # Save the data to a pickle file
                with open('Uniformer_Features.pkl', 'wb') as f:
                    pickle.dump(dataset, f)
        else:
            continue

    print("All features extracted successfully")
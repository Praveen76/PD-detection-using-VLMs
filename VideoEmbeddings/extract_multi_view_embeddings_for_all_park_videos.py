import torch
import click
import numpy as np
import os
import av
import time
import warnings
import pandas as pd
import pickle
import sys

from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, VivitImageProcessor
from typing import List, Dict, Union, Tuple

sys.path.append("/localdisk1/PARK/park_vlm/VideoEmbeddings")
from model_setup import *

warnings.filterwarnings("ignore")
np.random.seed(0)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

def time_ms():
    return time.time()*1000

def read_video_pyav(container, indices, reduced_height):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
        reduced_height (`int`): Each frame height is reduced to this height, and width is adjusted maintaining the original aspect ratio
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            old_height, old_width = frame.height, frame.width
            new_height = reduced_height
            downsample_ratio = new_height/old_height
            new_width = int(downsample_ratio*old_width)
            frame = frame.reformat(width=new_width, height=new_height)
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(n_frames_to_sample, stride, n_total_frames, n_views=1):
    '''
    Sample a given number of frame indices from the video.
    Args:
        n_frames_to_sample (`int`): Total number of frames to sample.
        stride (`int`): Sample every n-th frame.
        n_total_frames (`int`): Maximum allowed index of sample's last frame.
        n_views (`int`): Number of views to sample from the video
    Returns:
        indices (`List[List[int]]`): n_views x List of sampled frame indices for each view
    '''
    if n_total_frames==0:
        raise Exception("no frame in the video")
    
    indices = []

    converted_len = int(n_frames_to_sample * stride) #C:64
    multi_view_converted_len = converted_len*n_views #C: 64
    # max: 4 views, 2 strides, 32 frames = 4*2*32 = 256 frames = ~ 17 sec
    # avg: 4 views, 1 stride, 32 frames = 4*1*32 = 128 frames = ~ 8.5 sec

    if multi_view_converted_len<=n_total_frames:
        # we can sample all multi-views
        # if we have some extra frames, we can start from a random index, and consecutively sample multi views
        end_idx = np.random.randint(multi_view_converted_len, n_total_frames+1)

        start_idx = end_idx - multi_view_converted_len

        while start_idx<end_idx:
            # sample n frames uniformly out of converted_len frames, starting from start_idx 
            local_end_index = start_idx+converted_len
            view_indices = np.linspace(start_idx, local_end_index, num=n_frames_to_sample+1)[:-1].astype(np.int64)
            indices.append(view_indices)
            start_idx = local_end_index
    
    else:
        # we have less frames than required. So, overlapping would be necessary
        # minimum number of frames required for a single view (worst case): 1 view x 2 stride x 32 frames = 64 frames = ~4.3 sec
        if n_total_frames<converted_len:
            raise Exception("do not have enough frames for a single view")
        else:
            end_idx = n_total_frames
            start_idx = 0

            while len(indices)<n_views:
                local_end_index = start_idx+converted_len
                view_indices = np.linspace(start_idx, local_end_index, num=n_frames_to_sample+1)[:-1].astype(np.int64)
                indices.append(view_indices)

                if (local_end_index+converted_len)<n_total_frames:
                    start_idx = local_end_index
                else:
                    start_idx = np.random.randint(start_idx, (n_total_frames-converted_len)+1)
            # end of view collection
    return indices

@click.command()
@click.option("--model_tag", default="ViViT", help="Options: ViViT, TimeSformer, VideoMAE")
@click.option("--stride", default=2, help="Options: 1, 2, 4")
@click.option("--num_views", default=4, help="Options: 1, 2, 4")
def main(**cfg):
    # These parameters will be eventually obtained through command line arguments
    model_tag = cfg["model_tag"]
    completed_count = 0
    dataset = []

    count_filename = f'/localdisk1/PARK/park_vlm/VideoEmbeddings/{model_tag}/{model_tag}_{cfg["num_views"]}views_{cfg["stride"]}stride_Features_All_PARK_Videos_Count_Completed.txt'
    if os.path.exists(count_filename):
        print("Extraction already completed for some videos. Loading the already extracted features.")
        with open(count_filename, 'r') as f:
            x = f.readline().strip()
            completed_count = int(x)
            print(f"Completed count: {completed_count}")

        with open(f'/localdisk1/PARK/park_vlm/VideoEmbeddings/{model_tag}/{model_tag}_{cfg["num_views"]}views_{cfg["stride"]}stride_Features_All_PARK_Videos.pkl', 'rb') as f:
            dataset = pickle.load(f)
            print(dataset)
            print(f"Loaded embeddings: {len(dataset)}")
    
    count = 0

    # Load image processor and pre-trained video encoder model
    if model_tag=="ViViT":
        image_processor = VivitImageProcessor.from_pretrained(model_configs[model_tag]["model_name"])
    else:
        image_processor = AutoImageProcessor.from_pretrained(model_configs[model_tag]["model_name"])
    
    model = AutoModel.from_pretrained(model_configs[model_tag]["model_name"]).to(device)

    # process videos one by one
    for filename in tqdm(os.listdir("/localdisk2/park_videos/all_tasks/raw_videos/")):
        count +=1
        if count <= completed_count:
            continue

        try:
            item = {"filename":filename}
            file_path = os.path.join("/localdisk2/park_videos/all_tasks/raw_videos/", filename)
            container = av.open(file_path)

            n_total_frames=container.streams.video[0].frames

            if n_total_frames == 0:
                continue
                
            # sample n frames
            multi_view_indices = sample_frame_indices(n_frames_to_sample=model_configs[model_tag]["num_frames"], stride=cfg["stride"], n_total_frames=n_total_frames, n_views=cfg["num_views"])

            for view_idx in range(len(multi_view_indices)):
                indices = multi_view_indices[view_idx]    
                # pre-process video size (height and width), convert to numpy
                video = read_video_pyav(container, indices, reduced_height=model_configs[model_tag]["image_size"])
                # print(video.shape) # 16 x 224 x 298 x 3

                # prepare video for the model
                with torch.no_grad():
                    inputs = image_processor(list(video), return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    last_hidden_states = outputs.last_hidden_state
                    
                    mean_pooled = torch.mean(last_hidden_states, dim=-2).reshape(-1)    # (768,)
                    max_pooled = torch.max(last_hidden_states, dim=-2)[0].reshape(-1)   # (768,)
                    item[f"view_{view_idx}_mean_pooled_embedding"] = mean_pooled.cpu()
                    item[f"view_{view_idx}_max_pooled_embedding"] = max_pooled.cpu()

            # done scanning all views        
            dataset.append(item)

        except Exception as e:
            print(e)
        
        if count%100==0:
            # Save the data to a pickle file
            with open(f'/localdisk1/PARK/park_vlm/VideoEmbeddings/{model_tag}/{model_tag}_{cfg["num_views"]}views_{cfg["stride"]}stride_Features_All_PARK_Videos.pkl', 'wb') as f:
                pickle.dump(dataset, f)

            with open(f'/localdisk1/PARK/park_vlm/VideoEmbeddings/{model_tag}/{model_tag}_{cfg["num_views"]}views_{cfg["stride"]}stride_Features_All_PARK_Videos_Count_Completed.txt', 'w') as f:
                f.write(f"{count}\n")

            print(f"Partial progress saved: {count}/40K (approx)")
            # end of partial saving

        # end of file processing

    #finished iterating all files
    print("Feature extraction complete...")

if __name__ == "__main__":
    main()
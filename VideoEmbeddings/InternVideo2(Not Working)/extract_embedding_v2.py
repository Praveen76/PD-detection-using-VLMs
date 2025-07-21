import torch
import numpy as np
import os
import av
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
import decord
from decord import VideoReader, cpu
from typing import List, Dict, Union, Tuple
import time
import warnings
import pandas as pd
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")

def time_ms():
    return time.time()*1000

sample_video_path = "/localdisk2/park_videos/expression_tasks/raw_videos/2024-08-03T19%3A31%3A37.522Z_tmxJ4nNf3egSUivLQuaZhlylLOe2_smile.mp4"

np.random.seed(0)

model_configs = {
        "TimeSformer": {
            "model_name": "facebook/timesformer-base-finetuned-k400",
            "num_frames": 8,
            "image_size": 224,
        },
        "VideoMAE": {
            "model_name": "MCG-NJU/videomae-base",
            "num_frames": 16,
            "image_size": 224
        },
        "InternVideo2": {
            "model_name": "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
            "num_frames": 16,
            "image_size": 224
        }
    }

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

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

def sample_frame_indices(n_frames_to_sample, stride, n_total_frames):
    '''
    Sample a given number of frame indices from the video.
    Args:
        n_frames_to_sample (`int`): Total number of frames to sample.
        stride (`int`): Sample every n-th frame.
        n_total_frames (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(n_frames_to_sample * stride)
    end_idx = np.random.randint(converted_len, n_total_frames)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=n_frames_to_sample)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = torch.Tensor(frames)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)
    print(frames)
    exit(0)
    

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

video_path = sample_video_path
# sample uniformly 8 frames from the video
# video_tensor = load_video(video_path, num_segments=8, return_msg=False)
# video_tensor = video_tensor.to(device)

if __name__ == "__main__":
    # These parameters will be eventually obtained through command line arguments
    model_tag = "TimeSformer"

    smile_files = pd.read_csv("/localdisk1/PARK/park_multitask_fusion/data/facial_expression_smile/facial_dataset.csv")
    finger_files = pd.read_csv("/localdisk1/PARK/park_multitask_fusion/data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
    
    smile_files = list(smile_files["Filename"])
    finger_files = list(finger_files["filename"])

    dataset = []
    count = 0

    for filename in tqdm(os.listdir("/localdisk2/park_videos/all_tasks/raw_videos/")):
        try:
            if (filename in smile_files) or (filename in finger_files):
                item = {"filename":filename}
                file_path = os.path.join("/localdisk2/park_videos/all_tasks/raw_videos/", filename)
                container = av.open(file_path)
                
                video_tensor = load_video(file_path, num_segments=8, return_msg=False)

                model = AutoModel.from_pretrained(model_configs[model_tag]["model_name"]).to(device)

                # prepare video for the model
                with torch.no_grad():
                    inputs = video_tensor.to(device)

                    # t1 = time_ms()
                    outputs = model(**inputs)
                    last_hidden_states = outputs.last_hidden_state
                    # print(last_hidden_states.shape) # 1 x 1568 x 768
                    # t2 = time_ms()
                    # print(f"Runtime: {t2-t1} ms")

                mean_pooled = torch.mean(last_hidden_states, dim=-2).reshape(-1)    # (768,)
                max_pooled = torch.max(last_hidden_states, dim=-2)[0].reshape(-1)   # (768,)
                item["mean_pooled_embedding"] = mean_pooled.cpu()
                item["max_pooled_embedding"] = max_pooled.cpu()
                dataset.append(item)
                count +=1
                
                if count%50==0:
                    # Save the data to a pickle file
                    with open('TimeSformer_Features.pkl', 'wb') as f:
                        pickle.dump(dataset, f)

                    # with open('TimeSformer_Features.pkl', 'rb') as f:
                    #     loaded_data = pickle.load(f)
                    # print(loaded_data)

                    # exit(0)
            else:
                continue
        except Exception as e:
            print(e)
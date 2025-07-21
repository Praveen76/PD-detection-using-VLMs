import os
import sys
from split_ids import *
import torch
import pandas as pd

METADATA_FILE_PATH = "/localdisk1/PARK/park_vlm/Metadata/all_file_user_metadata.csv"
RAW_VIDEOS_PATH = {
    "all_tasks": "/localdisk2/park_videos/all_tasks",
    "facial_expressions": "/localdisk2/park_videos/expression_tasks",
    "free_flow_speech": "/localdisk2/park_videos/free_flow_speech",
    "smile": "/localdisk2/park_videos/expression_tasks",
    "speech": "/localdisk2/park_videos/all_tasks",
    "finger_tapping": "/localdisk2/park_videos/all_tasks"
}

dev_ids = get_dev_ids()
test_ids = get_test_ids()

label_mappings = {
    "no": 0,
    "yes": 1,
    "Unlikely": 0,
    "Possible": 1,
    "Probable": 1
}

def get_file_paths_and_labels(dataset_name="all_tasks"):
    '''
    {
    `train`: [
        {
            'unique_id': xxx,
            'file_name': xxx,
            'task': xxx,
            'file_path': xxx,
            'label': 0/1
        }
    ],
    'dev': [],
    'test': []
    }
    '''
    dataset = {}

    filenames_with_metadata = set()
    filenames_with_videos = set()
    
    df_videos = []
    DATA_DIR = os.path.join(RAW_VIDEOS_PATH[dataset_name], "raw_videos")
    uid = 1
    for filename in os.listdir(DATA_DIR):
        if (dataset_name == "smile") and ("smile" not in filename.lower()):
            continue

        if (dataset_name == "speech") and ("quick_brown_fox" not in filename.lower()):
            continue

        if (dataset_name == "finger_tapping") and ("finger_tapping" not in filename.lower()):
            continue

        data = {}
        data["file_name"] = filename
        data["file_path"] = os.path.join(DATA_DIR, filename)
        data["unique_id"] = f"{dataset_name}_{uid}"
        df_videos.append(data)
        uid +=1

    df_videos = pd.DataFrame.from_dict(df_videos) #unique_id, file_name, file_path
    df_videos["file_name"] = df_videos["file_name"].astype(str)
    # print(df_videos.columns) # Index(['file_name', 'file_path', 'unique_id'], dtype='object')
    # print(f"Number of files with videos: {len(df_videos)}")
    # print(df_videos.dtypes)

    # read metadata
    df_metadata = pd.read_csv(METADATA_FILE_PATH) #file_name, task, label
    df_metadata = df_metadata.rename(columns={
        "Filename": "file_name", 
        "Participant_ID": "pid", 
        "Task": "task",
        "Protocol": "protocol",
        "pd": "label"
    })
    df_metadata = df_metadata[["file_name", "pid", "task", "protocol", "label"]]
    df_metadata["file_name"] = df_metadata["file_name"].astype(str)
    # print(df_metadata.columns) Index(['file_name', 'pid', 'task', 'protocol', 'label'], dtype='object')
    # print(f"Number of files with metadata: {len(df_metadata)}")
    # print(df_metadata.dtypes)

    df_metadata = df_metadata.set_index('file_name')
    df_videos = df_videos.set_index('file_name')
    df = df_metadata.join(df_videos, on="file_name", how="inner")
    df = df.reset_index()

    df = df[~df["label"].isna()]

    # print(df.columns)
    print(f"Number of files with metadata and raw videos: {len(df)}")

    df["label"] = df["label"].apply(lambda x: label_mappings[x])
    
    dataset["dev"] = df[df["pid"].isin(dev_ids)]
    dataset["test"] = df[df["pid"].isin(test_ids)]
    dataset["train"] = df[~df["pid"].isin(list(set(dev_ids).union(set(test_ids))))]
    
    return dataset

if __name__ == "__main__":
    print("Facial Expression Dataset")
    dataset = get_file_paths_and_labels(dataset_name="facial_expressions")
    print(f"Size of training set: {len(dataset['train'])} videos, {len(dataset['train']['pid'].unique())} participants")
    print(f"Size of validation set: {len(dataset['dev'])} videos, {len(dataset['dev']['pid'].unique())} participants")
    print(f"Size of test set: {len(dataset['test'])} videos, {len(dataset['test']['pid'].unique())} participants")

    print("*"*20)

    print("Free-flow-speech Dataset")
    dataset = get_file_paths_and_labels(dataset_name="free_flow_speech")
    print(f"Size of training set: {len(dataset['train'])} videos, {len(dataset['train']['pid'].unique())} participants")
    print(f"Size of validation set: {len(dataset['dev'])} videos, {len(dataset['dev']['pid'].unique())} participants")
    print(f"Size of test set: {len(dataset['test'])} videos, {len(dataset['test']['pid'].unique())} participants")

    print("*"*20)

    print("All tasks Dataset")
    dataset = get_file_paths_and_labels(dataset_name="all_tasks")
    print(f"Size of training set: {len(dataset['train'])} videos, {len(dataset['train']['pid'].unique())} participants")
    print(f"Size of validation set: {len(dataset['dev'])} videos, {len(dataset['dev']['pid'].unique())} participants")
    print(f"Size of test set: {len(dataset['test'])} videos, {len(dataset['test']['pid'].unique())} participants")

    print(dataset["test"].iloc[0]["file_path"])

    print("Smile Dataset")
    dataset = get_file_paths_and_labels(dataset_name="smile")
    print(f"Size of training set: {len(dataset['train'])} videos, {len(dataset['train']['pid'].unique())} participants")
    print(f"Size of validation set: {len(dataset['dev'])} videos, {len(dataset['dev']['pid'].unique())} participants")
    print(f"Size of test set: {len(dataset['test'])} videos, {len(dataset['test']['pid'].unique())} participants")
    print(dataset["test"].iloc[0]["file_path"])

    print("Speech Dataset")
    dataset = get_file_paths_and_labels(dataset_name="speech")
    print(f"Size of training set: {len(dataset['train'])} videos, {len(dataset['train']['pid'].unique())} participants")
    print(f"Size of validation set: {len(dataset['dev'])} videos, {len(dataset['dev']['pid'].unique())} participants")
    print(f"Size of test set: {len(dataset['test'])} videos, {len(dataset['test']['pid'].unique())} participants")
    print(dataset["test"].iloc[0]["file_path"])

    print("Finger-tapping Dataset")
    dataset = get_file_paths_and_labels(dataset_name="finger_tapping")
    print(f"Size of training set: {len(dataset['train'])} videos, {len(dataset['train']['pid'].unique())} participants")
    print(f"Size of validation set: {len(dataset['dev'])} videos, {len(dataset['dev']['pid'].unique())} participants")
    print(f"Size of test set: {len(dataset['test'])} videos, {len(dataset['test']['pid'].unique())} participants")
    print(dataset["test"].iloc[0]["file_path"])
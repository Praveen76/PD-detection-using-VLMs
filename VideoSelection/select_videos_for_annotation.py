import os
import pandas as pd
import numpy as np
import random

# Random behavior -- do not re-run after finalizing
# exit(0)

# Step 1: read all the available filenames of the videos we have.
VIDEOS_ROOT = "/localdisk1/PARK/park_vlm/Videos"
filenames = []

for x in os.listdir(VIDEOS_ROOT):
    filenames.append(x)

all_filenames = set(filenames)
print(f"Total number of files: {len(all_filenames)}") # there seem to be 575 files in total

# Step 2: read all the filenames with available metadata.
METADATA_FILE = "/localdisk1/PARK/park_vlm/Metadata/all_file_user_metadata.csv"
metadata_df = pd.read_csv(METADATA_FILE)
filenames_with_metadata = set(metadata_df["Filename"]).intersection(all_filenames)
print(f"Number of files with metadata: {len(filenames_with_metadata)}")

# Step 3: find files that do not have metadata.
files_without_metadata = all_filenames.difference(filenames_with_metadata)
print(f"Number of files without metadata: {len(files_without_metadata)}")
print(files_without_metadata) # one video does not have metadata, but this is a good free-flow-speech.

# Step 4: look for annotation information.
ANNOTATION_LOCS = {"nami": "/localdisk1/PARK/park_vlm/Annotations/Clinical/Nami", 
                   "natalia": "/localdisk1/PARK/park_vlm/Annotations/Clinical/Natalia",
                   "cayla": "/localdisk1/PARK/park_vlm/Annotations/Clinical/Cayla",
                   "taylor": "/localdisk1/PARK/park_vlm/Annotations/Non-clinical/Taylor/free_flow_speech"}

# 4a. populate clinical annotations
annotated_files = {"nami": [], "natalia": [], "cayla": [], "taylor": []}

for annotator in ["nami", "natalia", "cayla"]:
    common_annotation_file = os.path.join(ANNOTATION_LOCS[annotator], f"{annotator}-common-25-1.csv")
    if not os.path.exists(common_annotation_file):
        continue

    common_df = pd.read_csv(common_annotation_file)
    common_df["filename"] = common_df["video"].apply(lambda x: x.split("/")[-1])
    annotated_files[annotator].extend(common_df["filename"])

    unique_annotation_file = os.path.join(ANNOTATION_LOCS[annotator], f"{annotator}-unique-100-1.csv")
    if not os.path.exists(unique_annotation_file):
        continue
    unique_df = pd.read_csv(unique_annotation_file)
    unique_df["filename"] = unique_df["video"].apply(lambda x: x.split("/")[-1])
    annotated_files[annotator].extend(unique_df["filename"])

for x in os.listdir(ANNOTATION_LOCS["taylor"]):
    filename = x[:-4]+".mp4"
    annotated_files["taylor"].append(filename)

for annotator in ["nami", "natalia", "cayla", "taylor"]:
    print(f"Number of files annotated by {annotator}: {len(set(annotated_files[annotator]))}")


common_files = set(annotated_files["nami"]).intersection(set(annotated_files["natalia"]))
# it appears Nami and Natalia only have 2 files in common

common_files = set(annotated_files["nami"]).intersection(set(annotated_files["taylor"]))
# Taylor and Nami has 61 in common

annotations = []
for filename in filenames_with_metadata:
    annotation_dict = {"filename":filename, "nami":0, "natalia":0, "cayla":0, "taylor":0, "is_free_flow_speech":False}
    for annotator in ["nami", "natalia", "cayla", "taylor"]:
        if filename in annotated_files[annotator]:
            annotation_dict[annotator] = 1
    annotations.append(annotation_dict)
annotations_df = pd.DataFrame.from_dict(annotations)

annotations_df["is_common"] = annotations_df.apply(lambda x: (x["nami"]+x["natalia"])==2, axis=1)
annotations_df["is_not_annotated"] = annotations_df.apply(lambda x: (x["nami"]+x["natalia"]+x["taylor"])==0, axis=1)
annotations_df.to_csv("/localdisk1/PARK/park_vlm/Annotations/annotation_summary.csv", index=False)

# get 12 files that are annotated by natalia but not by nami
subset_df = annotations_df[(annotations_df["nami"]==0) & (annotations_df["natalia"]==1)]
filenames = set(subset_df["filename"])
selected_files_for_nami = random.sample(sorted(filenames), 12)

# get 11 files that are annotated by nami but not nataliasubset_df = annotations_df[(annotations_df["nami"]==0) & (annotations_df["natalia"]==1)]
subset_df = annotations_df[(annotations_df["nami"]==1) & (annotations_df["natalia"]==0)]
filenames = set(subset_df["filename"])
selected_files_for_natalia = random.sample(sorted(filenames), 11)
print("Selected files for Natalia:")
print(selected_files_for_natalia)

# setup 25 for Cayla so that there are 25 common for all
selected_files_for_cayla_common = []
selected_files_for_cayla_common.extend(annotations_df[annotations_df["is_common"]]["filename"])
selected_files_for_cayla_common.extend(selected_files_for_nami)
selected_files_for_cayla_common.extend(selected_files_for_natalia)
print("Selected common files for Cayla:")
print(selected_files_for_cayla_common)

# setup 100 for Cayla which are not annotated by Taylor or Nami or Natalia
subset_df = annotations_df[annotations_df["is_not_annotated"]]
filenames = set(subset_df["filename"])
selected_files_for_cayla_unique = random.sample(sorted(filenames), 100)
print("Selected unique files for Cayla:")
print(selected_files_for_cayla_unique)

with open("remaining_annotations.txt","w") as f:
    f.write("Selected files for Nami:\n")
    for filename in selected_files_for_nami:
        f.write(filename+"\n")
    f.write("\n")
    f.write("\n")

    f.write("Selected files for Natalia:\n")
    for filename in selected_files_for_natalia:
        f.write(filename+"\n")
    f.write("\n")
    f.write("\n")

    f.write("Selected common files for Cayla:\n")
    for filename in selected_files_for_cayla_common:
        f.write(filename+"\n")
    f.write("\n")
    f.write("\n")

    f.write("Selected unique files for Cayla:\n")
    for filename in selected_files_for_cayla_unique:
        f.write(filename+"\n")
    f.write("\n")
    f.write("\n")
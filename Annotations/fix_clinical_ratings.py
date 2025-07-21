'''
The common vs unique ratings by the clinicians were mixed up before.
The script fixes these issues and generates 
    [NAME]_common_final.csv
    [NAME]_unique_final.csv 
for each annotator.
'''
import pandas as pd
import os

def extract_video_name(video_url):
    """
    The URL is the link to GCP/LabelStudio bucket
    We need the filename from it
    """
    return video_url.split("/")[-1]

# CSV files where clinical annotations are saved
cayla_common = "/localdisk1/PARK/park_vlm/Annotations/Clinical/Cayla/cayla-common-25-1.csv"
cayla_unique = "/localdisk1/PARK/park_vlm/Annotations/Clinical/Cayla/cayla-unique-100-1.csv"
nami_common_1 = "/localdisk1/PARK/park_vlm/Annotations/Clinical/Nami/nami-common-25-1.csv"
nami_common_2 = "/localdisk1/PARK/park_vlm/Annotations/Clinical/Nami/nami-common-25-2.csv"
nami_unique_1 = "/localdisk1/PARK/park_vlm/Annotations/Clinical/Nami/nami-unique-100-1.csv"
natalia_common_1 = "/localdisk1/PARK/park_vlm/Annotations/Clinical/Natalia/natalia-common-25-1.csv"
natalia_common_2 = "/localdisk1/PARK/park_vlm/Annotations/Clinical/Natalia/natalia-common-25-2.csv"
natalia_unique_1 = "/localdisk1/PARK/park_vlm/Annotations/Clinical/Natalia/natalia-unique-100-1.csv"

'''
Combine all the annotations done by Nami
'''
df_nami = pd.read_csv(nami_common_1)
df_nami_additional = pd.read_csv(nami_common_2)
df_nami = pd.concat([df_nami, df_nami_additional], ignore_index=True)
df_nami_additional = pd.read_csv(nami_unique_1)
df_nami = pd.concat([df_nami, df_nami_additional], ignore_index=True)
# Standarize video file names for joining with other dataframes
df_nami["video"] = df_nami["video"].apply(extract_video_name)

'''
Combine all the annotations done by Natalia
'''
df_natalia = pd.read_csv(natalia_common_1)
df_natalia_additional = pd.read_csv(natalia_common_2)
df_natalia = pd.concat([df_natalia, df_natalia_additional], ignore_index=True)
df_natalia_additional = pd.read_csv(natalia_unique_1)
df_natalia = pd.concat([df_natalia, df_natalia_additional], ignore_index=True)
# Standarize video file names for joining with other dataframes
df_natalia["video"] = df_natalia["video"].apply(extract_video_name)

'''
Fetch the common annotations done by Cayla
'''
df = pd.read_csv(cayla_common)
df["video"] = df["video"].apply(extract_video_name)

'''
Find videos that are common among all annotators
'''
common_videos = set(df_nami["video"]).intersection(set(df["video"]))
common_videos = set(df_natalia["video"]).intersection(df_nami["video"]).intersection(set(df["video"]))
print(f"Number of common videos: {len(common_videos)}")

'''
Construct dataframe for the common videos
'''
# for Nami
df_nami_common = df_nami[df_nami["video"].isin(common_videos)]
df_nami_common.to_csv(os.path.join("/localdisk1/PARK/park_vlm/Annotations/Clinical/Nami", "nami_common_final.csv"), index=False)

# for Natalia
df_natalia_common = df_natalia[df_natalia["video"].isin(common_videos)]
df_natalia_common.to_csv(os.path.join("/localdisk1/PARK/park_vlm/Annotations/Clinical/Natalia", "natalia_common_final.csv"), index=False)

# for Cayla
df.to_csv(os.path.join("/localdisk1/PARK/park_vlm/Annotations/Clinical/Cayla", "cayla_common_final.csv"), index=False)

'''
Construct dataframe for the unique videos
'''
# Nami
df_nami_unique = df_nami[~(df_nami["video"].isin(common_videos))]
df_nami_unique.to_csv(os.path.join("/localdisk1/PARK/park_vlm/Annotations/Clinical/Nami", "nami_unique_final.csv"), index=False)

# Natalia
df_natalia_unique = df_natalia[~(df_natalia["video"].isin(common_videos))]
df_natalia_unique.to_csv(os.path.join("/localdisk1/PARK/park_vlm/Annotations/Clinical/Natalia", "natalia_unique_final.csv"), index=False)

# Cayla
df_cayla_unique = pd.read_csv(cayla_unique)
df_cayla_unique.to_csv(os.path.join("/localdisk1/PARK/park_vlm/Annotations/Clinical/Cayla", "cayla_unique_final.csv"), index=False)
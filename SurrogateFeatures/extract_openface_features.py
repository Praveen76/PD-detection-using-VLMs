import pandas as pd 
import numpy as np 
import cv2
import mediapipe as mp
from tqdm import tqdm

from protobuf_to_dict import protobuf_to_dict
import os
import sys
from scipy.fftpack import fft

import click 

fps = 15

# # Redirect STDERR to null
# sys.stderr = open(os.devnull, 'w')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

def calculate_blink_features(df, au_column_activation = 'AU45_c', au_column_intensity = 'AU45_r', fps = 15, long_blink_threshold = 0.2):
    """Calcualte features based on AU45 for blink dynamics"""
    # Blink Count
    blink_count = 0
    
    for i in range(1, len(df)):
        # Detect a blink by finding a transition from 0 to 1
        if df[au_column_activation].iloc[i] == 1 and df[au_column_activation].iloc[i - 1] == 0:
            blink_count += 1
    
    # Blink Frequency
    duration_of_video = len(df) / fps
    blink_frequency = blink_count / duration_of_video
    
    # Blink Duration
    blink_duration = 0
    blink_duration_list = []
    
    for i in range(len(df)):
        if df[au_column_activation][i] == 1:
            blink_duration += 1
        else:
            if blink_duration > 0:
                blink_duration_list.append(blink_duration)
                blink_duration = 0
    if blink_duration > 0:
        blink_duration_list.append(blink_duration)
            
    # Convert the blink duration to seconds
    blink_duration_list = [duration / fps for duration in blink_duration_list]
        
    # Blink Duration Min, Max, Mean, and Std
    if len(blink_duration_list) > 0:
        blink_duration_mean = np.mean(blink_duration_list)
        blink_duration_std = np.std(blink_duration_list)
        blink_duration_min = np.min(blink_duration_list)
        blink_duration_max = np.max(blink_duration_list)
    else:
        blink_duration_mean = 0
        blink_duration_std = 0
        blink_duration_min = 0
        blink_duration_max = 0

    
    # Inter Blink Interval (IBI)

    # a blink means 1s in the AU45 column for a few consecutive frames
    # we will measure the time between the end of a blink and the start of the next blink
    # this will give us the inter blink interval
    
    ibi_list = []
    last_frame = 0
    first_blink_passed = False
    last_blink_end = 0
    
    for i in range(len(df)):
        current_frame = df[au_column_activation].iloc[i]
        if last_frame == 0 and current_frame == 0:
            continue
        elif last_frame == 1 and current_frame == 1:
            last_blink_end = i
        elif last_frame == 1 and current_frame == 0:
            if not first_blink_passed:
                first_blink_passed = True
            last_blink_end = i
        elif last_frame == 0 and current_frame == 1:
            if first_blink_passed:
                ibi = i - last_blink_end
                ibi_list.append(ibi)
            last_blink_end = i
            
                
        last_frame = current_frame
    
    ibi_list = [ibi / fps for ibi in ibi_list]
    
    
    if len(ibi_list) > 0:
        ibi_mean = np.mean(ibi_list)
        ibi_std = np.std(ibi_list)
        ibi_min = np.min(ibi_list)
        ibi_max = np.max(ibi_list)
    else:
        ibi_mean = 0
        ibi_std = 0
        ibi_min = 0
        ibi_max = 0
    
    
    # Proportion of time spent blinking
    blink_duration_total = sum(blink_duration_list)
    total_duration = len(df) / fps
    blink_proportion = blink_duration_total / total_duration
    
    # Capture Long Blinks
    long_blinks = [duration for duration in blink_duration_list if duration > long_blink_threshold]
    long_blinks_proportion = len(long_blinks) / blink_count if blink_count > 0 else 0
    
    
    # Calcualte the blink Intensity using AU45_r from frames where the blink is detected
    blink_intensity_values = df[df[au_column_activation] == 1][au_column_intensity]
    
    # check for empty list
    if len(blink_intensity_values) > 0:
        blink_intensity_max = blink_intensity_values.max()
        blink_intensity_min = blink_intensity_values[blink_intensity_values > 0].min() if len(blink_intensity_values[blink_intensity_values > 0]) > 0 else 0
        blink_intensity_mean = blink_intensity_values.mean()
        blink_intensity_variability = blink_intensity_values.std()
    else:
        blink_intensity_max = 0
        blink_intensity_min = 0
        blink_intensity_mean = 0
        blink_intensity_variability = 0
        
    features = {
        'blink_count': blink_count,
        'blink_frequency': blink_frequency,
        'blink_duration_mean': blink_duration_mean,
        'blink_duration_variability': blink_duration_std,
        'blink_duration_min': blink_duration_min,
        'blink_duration_max': blink_duration_max,
        'ibi_mean': ibi_mean,
        'ibi_std': ibi_std,
        'ibi_min': ibi_min,
        'ibi_max': ibi_max,
        'blink_proportion': blink_proportion,
        'long_blinks_proportion': long_blinks_proportion,
        'blink_intensity_mean': blink_intensity_mean,
        'blink_intensity_variability': blink_intensity_variability,
        'blink_intensity_max': blink_intensity_max,
        'blink_intensity_min': blink_intensity_min
    }
    
    return features





def calculate_lips_movement_dynamics(df, fps = 15):
    """Calculate features based on AU12 for lips corner movement dynamics"""
    
    au_column_activation = 'AU12_c'
    au_column_intensity = 'AU12_r'
    
    # Frequency of AU12 Activation
    lips_puller_activation_frequency = df[au_column_activation].sum() / len(df)
    
    # Intensities of AU12 where it is activated
    # We will consider the intesity to be zero if the AU12 is not activated
    # We will consider only those frames where AU12 is activated
    lips_puller_intensity = df[df[au_column_activation] == 1][au_column_intensity]
    # lips_puller_intensity = df[au_column_intensity] * df[au_column_activation]
    
    if len(lips_puller_intensity) > 0:
        # Average Intensity of AU12
        lips_puller_intensity_mean = lips_puller_intensity.mean()
        # Variability of AU12 Intensity
        lips_puller_intensity_variability = lips_puller_intensity.std()
        # Take Min and Max Intensity of AU12, where it is activated (skip 0)
        lips_puller_intensity_min = lips_puller_intensity[lips_puller_intensity > 0].min() if len(lips_puller_intensity[lips_puller_intensity > 0]) > 0 else 0
        lips_puller_intensity_max = lips_puller_intensity.max() 
    
    else:
        lips_puller_intensity_mean = 0
        lips_puller_intensity_variability = 0
        lips_puller_intensity_min = 0
        lips_puller_intensity_max = 0
        
    """Calculate features based on AU14 for lips tightening dynamics"""
    
    au_column_activation = 'AU14_c'
    au_column_intensity = 'AU14_r'
    
    # Frequency of frames where AU14 is activated
    lips_tightening_activation_frequency = df[au_column_activation].sum() / len(df)
    
    # Intensity of AU14 where it is activated
    lips_tightening_intensity = df[df[au_column_activation] == 1][au_column_intensity]
    # lips_tightening_intensity = df[au_column_intensity] * df[au_column_activation]
    
    if len(lips_tightening_intensity) > 0:    
        # Mean Intensity of AU14
        lips_tightening_intensity_mean = lips_tightening_intensity.mean()
        # Minimum and Maximum Intensity of AU14
        lips_tightening_intensity_min = lips_tightening_intensity[lips_tightening_intensity > 0].min() if len(lips_tightening_intensity[lips_tightening_intensity > 0]) > 0 else 0
        lips_tightening_intensity_max = lips_tightening_intensity.max()
        # Variability of AU14 Intensity
        lips_tightening_intensity_variability = lips_tightening_intensity.std()
    else:
        lips_tightening_intensity_mean = 0
        lips_tightening_intensity_variability = 0
        lips_tightening_intensity_min = 0
        lips_tightening_intensity_max = 0
    
    
    
    """Calculate features based on AU25 for lips parting dynamics"""
    
    au_column_activation = 'AU25_c'
    au_column_intensity = 'AU25_r'
    
    # Frequency of frames where AU25 is activated
    lips_parting_activation_frequency = df[au_column_activation].sum() / len(df)
    
    # Lips Parting Duration 
    parting_durations = []
    parting_duration = 0
    
    for i in range(len(df)):
        if df[au_column_activation][i] == 1:
            parting_duration += 1
        else:
            if parting_duration > 0:
                parting_durations.append(parting_duration)
                parting_duration = 0
                
    if parting_duration > 0:
        parting_durations.append(parting_duration)
        
    parting_durations = [duration / fps for duration in parting_durations]
    
    # Lips Parting Duration Mean, Min, Max, and Variability
    if len(parting_durations) > 0:
        lips_parting_duration_mean = np.mean(parting_durations)
        lips_parting_duration_std = np.std(parting_durations)
        lips_parting_duration_min = np.min(parting_durations)
        lips_parting_duration_max = np.max(parting_durations)
    else:
        lips_parting_duration_mean = 0
        lips_parting_duration_std = 0
        lips_parting_duration_min = 0
        lips_parting_duration_max = 0
    
    # Lips Parting Intensity
    lips_parting_intensity = df[df[au_column_activation] == 1][au_column_intensity]
    # lips_parting_intensity = df[au_column_intensity] * df[au_column_activation]
    
    # Lips Parting Intensity Mean, Min, Max, and Variability
    if len(lips_parting_intensity) > 0:
        lips_parting_intensity_mean = lips_parting_intensity.mean()
        lips_parting_intensity_variability = lips_parting_intensity.std()
        lips_parting_intensity_min = lips_parting_intensity[lips_parting_intensity > 0].min() if len(lips_parting_intensity[lips_parting_intensity > 0]) > 0 else 0
        lips_parting_intensity_max = lips_parting_intensity.max()
    else:
        lips_parting_intensity_mean = 0
        lips_parting_intensity_variability = 0
        lips_parting_intensity_min = 0
        lips_parting_intensity_max = 0
    
    features = {
        'lips_puller_activation_frequency': lips_puller_activation_frequency,
        'lips_puller_intensity_mean': lips_puller_intensity_mean,
        'lips_puller_intensity_variability': lips_puller_intensity_variability,
        'lips_puller_intensity_min': lips_puller_intensity_min,
        'lips_puller_intensity_max': lips_puller_intensity_max,
        'lips_tightening_activation_frequency': lips_tightening_activation_frequency,
        'lips_tightening_intensity_mean': lips_tightening_intensity_mean,
        'lips_tightening_intensity_variability': lips_tightening_intensity_variability,
        'lips_tightening_intensity_min': lips_tightening_intensity_min,
        'lips_tightening_intensity_max': lips_tightening_intensity_max,
        'lips_parting_activation_frequency': lips_parting_activation_frequency,
        'lips_parting_duration_mean': lips_parting_duration_mean,
        'lips_parting_duration_variability': lips_parting_duration_std,
        'lips_parting_duration_min': lips_parting_duration_min,
        'lips_parting_duration_max': lips_parting_duration_max,
        'lips_parting_intensity_mean': lips_parting_intensity_mean,
        'lips_parting_intensity_variability': lips_parting_intensity_variability,
        'lips_parting_intensity_min': lips_parting_intensity_min,
        'lips_parting_intensity_max': lips_parting_intensity_max
    }  
    
    return features



def calculate_jaw_movement_dynamics(df, fps=15):
    """Calculate features based on AU26 for jaw drop dynamics"""
    
    au_column_activation = 'AU26_c'
    au_column_intensity = 'AU26_r'
    
    # Frequency of frames where AU26 is activated
    jaw_drop_activation_frequency = df[au_column_activation].sum() / len(df)
    
    # Time spent with jaw drop
    jaw_drop_durations = []
    jaw_drop_duration = 0
    
    for i in range(len(df)):
        if df[au_column_activation][i] == 1:
            jaw_drop_duration += 1
        else:
            if jaw_drop_duration > 0:
                jaw_drop_durations.append(jaw_drop_duration)
                jaw_drop_duration = 0
                
    if jaw_drop_duration > 0:
        jaw_drop_durations.append(jaw_drop_duration)
        
    jaw_drop_durations = [duration / fps for duration in jaw_drop_durations]
    
    # Jaw Drop Duration Mean, Min, Max, and Variability
    if len(jaw_drop_durations) > 0:
        jaw_drop_duration_mean = np.mean(jaw_drop_durations)
        jaw_drop_duration_std = np.std(jaw_drop_durations)
        jaw_drop_duration_min = np.min(jaw_drop_durations)
        jaw_drop_duration_max = np.max(jaw_drop_durations)
        
    else:
        jaw_drop_duration_mean = 0
        jaw_drop_duration_std = 0
        jaw_drop_duration_min = 0
        jaw_drop_duration_max = 0
        
    
    # Intensity of AU26 where it is activated
    jaw_drop_intensity = df[df[au_column_activation] == 1][au_column_intensity]
    # jaw_drop_intensity = df[au_column_intensity] * df[au_column_activation]
    
    if len(jaw_drop_intensity) > 0:
        # Mean Intensity of AU26
        jaw_drop_intensity_mean = jaw_drop_intensity.mean()
        
        # Minimum and Maximum Intensity of AU26
        jaw_drop_intensity_min = jaw_drop_intensity[jaw_drop_intensity > 0].min() if len(jaw_drop_intensity[jaw_drop_intensity > 0]) > 0 else 0
        jaw_drop_intensity_max = jaw_drop_intensity.max()
        
        # Variability of AU26 Intensity
        jaw_drop_intensity_variability = jaw_drop_intensity.std()
        
    else:
        jaw_drop_intensity_mean = 0
        jaw_drop_intensity_variability = 0
        jaw_drop_intensity_min = 0
        jaw_drop_intensity_max = 0
    
    features = {
        'jaw_drop_activation_frequency': jaw_drop_activation_frequency,
        'jaw_drop_duration_mean': jaw_drop_duration_mean,
        'jaw_drop_duration_variability': jaw_drop_duration_std,
        'jaw_drop_duration_min': jaw_drop_duration_min,
        'jaw_drop_duration_max': jaw_drop_duration_max,
        'jaw_drop_intensity_mean': jaw_drop_intensity_mean,
        'jaw_drop_intensity_variability': jaw_drop_intensity_variability,
        'jaw_drop_intensity_min': jaw_drop_intensity_min,
        'jaw_drop_intensity_max': jaw_drop_intensity_max
    }
    
    return features



def calculate_other_features(df, au_list = ['01', '05', '06', '07'], 
                             au_names = ['inner_brow_raiser', 'upper_lid_raiser', 'cheek_raiser', 'lid_tightener'], fps=15):
    """Calculate features based on other AUs"""
    
    features = {}
    
    for i in range(len(au_list)):
        au = au_list[i]
        au_column_activation = f'AU{au}_c'
        au_column_intensity = f'AU{au}_r'
        
        # Frequency of frames where AU is activated
        au_activation_frequency = df[au_column_activation].sum() / len(df)
        
        # Intensity of AU where it is activated
        au_intensity = df[df[au_column_activation] == 1][au_column_intensity]
        # au_intensity = df[au_column_intensity] * df[au_column_activation]
        
        if len(au_intensity) > 0:
        
            # Mean Intensity of AU
            au_intensity_mean = au_intensity.mean() if len(au_intensity) > 0 else 0
            
            # Minimum and Maximum Intensity of AU
            au_intensity_min = au_intensity[au_intensity > 0].min() if len(au_intensity[au_intensity > 0]) > 0 else 0
            
            au_intensity_max = au_intensity.max()
            
            # Variability of AU Intensity
            au_intensity_variability = au_intensity.std()
            
        else:
            au_intensity_mean = 0
            au_intensity_variability = 0
            au_intensity_min = 0
            au_intensity_max = 0
        
        features[f'{au_names[i]}_activation_frequency'] = au_activation_frequency
        features[f'{au_names[i]}_intensity_mean'] = au_intensity_mean
        features[f'{au_names[i]}_intensity_variability'] = au_intensity_variability
        features[f'{au_names[i]}_intensity_min'] = au_intensity_min
        features[f'{au_names[i]}_intensity_max'] = au_intensity_max
        
        
    return features



@click.command()
@click.option("--subdirectory",default='finger_tapping')
def main(**args):
    
    failed_videos = []
    
    root_directory = "/Users/tariqadnan/Downloads/OpenFace_extracts"
    
    subdirectories = os.listdir(root_directory)
    
    for subdirectory in subdirectories:
        
        subdirectory_path = os.path.join(root_directory, subdirectory)
        
        full_features_df = pd.DataFrame()
        
        print("Processing subdirectory: ", subdirectory_path)
        if os.path.isdir(subdirectory_path):
            list_of_files = os.listdir(subdirectory_path)
            # print("list of files in the subdirectory: ", list_of_files)
            print("Number of files in the subdirectory: ", len(list_of_files))
            for i in tqdm(range(len(list_of_files)), desc="Processing files"):
                file = list_of_files[i]
                if file.endswith('.csv'):
                    file_path = os.path.join(subdirectory_path, file)
                    openface_df = pd.read_csv(file_path)
                    # check if the file has at least 2*fps rows
                    if len(openface_df) < 2 * fps:
                        failed_videos.append(file)
                        continue
                    # trim the column names
                    openface_df.columns = openface_df.columns.str.strip()
                    features = {
                        'filename': file[:-4]
                    }
                    features.update(calculate_blink_features(openface_df))
                    features.update(calculate_lips_movement_dynamics(openface_df))
                    features.update(calculate_jaw_movement_dynamics(openface_df))
                    features.update(calculate_other_features(openface_df))
                    # make a dataframe from the features
                    df_features = pd.DataFrame([features])
                    full_features_df = pd.concat([full_features_df, df_features])
                    
            outpute_name = subdirectory + '_openface_features.csv'
            full_features_df.to_csv(outpute_name, index=False)
            print("Features saved to: ", outpute_name)
            print("Writing failed videos to file...")
            with open("failed_videos.txt", "w") as file:
                for video in failed_videos:
                    file.write(video + "\n")
        else:
            print("Subdirectory does not exist")
        
    
    

if __name__ == "__main__":
    print("Starting the process")
    main()


# filename = "2021-11-22T15%3A51%3A30.189Z_3ZqclU5mVqNRUb8TiZc8OXKbcZi1_smile.csv"

# df = pd.read_csv("Sample Data/openface extracts/" + filename)  # Load your data file here
# # trim the column names
# df.columns = df.columns.str.strip()

# features = {
#     'filename': filename[:-3]
# }
# features.update(calculate_blink_features(df))
# features.update(calculate_lips_movement_dynamics(df))
# features.update(calculate_jaw_movement_dynamics(df))
# features.update(calculate_other_features(df))

# # make a dataframe from the features
# df_features = pd.DataFrame([features])

# outpute_name = 'openface_featues.csv'
# df_features.to_csv("Sample Data/" + outpute_name, index=False)

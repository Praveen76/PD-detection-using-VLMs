import sys
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # or 'osmesa'

import pandas as pd 
import numpy as np 

import mediapipe as mp
from tqdm import tqdm
from protobuf_to_dict import protobuf_to_dict
import os


from scipy.fftpack import fft

import logging
import cv2

import click 

fps = 15

# Configure logging to suppress specific messages
logging.basicConfig(level=logging.ERROR)

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except AttributeError:
    pass  # Fallback for OpenCV versions without utils.logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def safe_divide(numerator, denominator, default_value=0.0):
    """
    Safely divides two numbers.
    
    Parameters:
    numerator: The value to be divided.
    denominator: The value to divide by.
    default_value: The value to return if the denominator is zero or invalid.

    Returns:
    The result of the division or the default value if the division is unsafe.
    """
    try:
        return numerator / denominator if denominator else default_value
    except ZeroDivisionError:
        return default_value
    except (TypeError, ValueError):
        return default_value

# we send reference points for normalization -- two outer points of the eyes

landmark_reference_points = {
    'reference_point1': '130',
    'reference_point2': '359'
}

def calculate_pairwise_distances(landmarks, point1, point2):
    distance = np.sqrt(
        (landmarks[f'{point1}_x'] - landmarks[f'{point2}_x'])**2 +
        (landmarks[f'{point1}_y'] - landmarks[f'{point2}_y'])**2
    )
    
    # normalize the distance with reference distance
    distance = distance / landmarks['reference_distance']

    
    return distance

# Now we caclulate blink related features using mediapipe landmarks

def calculate_ear(eye_landmarks, upper_lid, lower_lid, left_corner, right_corner):
    '''
    Calculate Eye Aspect Ratio (EAR) from eye landmarks
    
    Parameters:
    eye_landmarks (list): dataframe row containing eye landmarks
    upper_lid, lower_lid, left_corner, right_corner (str): indices for the eye landmarks
    
    Returns:
    ear: Eye Aspect Ratio
    '''
    
    # calculate the euclidean distance between the vertical eye landmarks
    vertical_dist = calculate_pairwise_distances(eye_landmarks, upper_lid, lower_lid)

    
    # calculate the euclidean distance between the horizontal eye landmarks
    horizontal_dist = calculate_pairwise_distances(eye_landmarks, left_corner, right_corner)
    
    # calculate the EAR
    #ear = vertical_dist / horizontal_dist
    ear = safe_divide(vertical_dist, horizontal_dist)
    
    return ear
    

def blink_features_landmaks(df_landmarks, fps = 15):
    '''
    Calculate blink realted features from MediaPipe landmarks
    
    Parameters:
    df_landmarks (pd.DataFrame): DataFrame containing mediapipe landmarks
    fps (int): Frame per second of the video
    
    Returns:
    blink_features_landmarks: Dictionary containing blink related features
    '''
    
    # indices of landmarks for left and right eye
    right_eye_landmarks = {
        'upper_lid': '159', 
        'lower_lid': '145', 
        'left_corner': '33', 
        'right_corner': '133'
    }
    
    left_eye_landmarks = {
        'upper_lid': '386', 
        'lower_lid': '374', 
        'left_corner': '362', 
        'right_corner': '263'
    }
        
    # we calculate EAR for each eyes on each frame
    df_landmarks['right_ear'] = df_landmarks.apply(lambda x: calculate_ear(x, **right_eye_landmarks), axis=1)
    df_landmarks['left_ear'] = df_landmarks.apply(lambda x: calculate_ear(x, **left_eye_landmarks), axis=1)
    
    # calculate the average EAR for both eyes
    df_landmarks['ear'] = (df_landmarks['right_ear'] + df_landmarks['left_ear']) / 2
    
    # threshold for blink detection
    ear_threshold = 0.2
    
    # detect blinks
    blink_count = 0
    for i in range(1, len(df_landmarks)):
        if df_landmarks['ear'].iloc[i] < ear_threshold and df_landmarks['ear'].iloc[i - 1] >= ear_threshold:
            blink_count += 1
            
    # calculate blink frequency
    duration_of_video = len(df_landmarks) / fps
    
    # blink_frequency = blink_count / duration_of_video
    blink_frequency = safe_divide(blink_count, duration_of_video)
    
    # we capture any asymmetry in the blink
    df_landmarks['ear_diff'] = np.abs(df_landmarks['right_ear'] - df_landmarks['left_ear'])
    
    # calculate the average difference in EAR
    ear_diff_mean = df_landmarks['ear_diff'].mean()
    
    # calculate the variability in the difference in EAR
    ear_diff_variability = df_landmarks['ear_diff'].std()
    
    # calculate the maximum difference in EAR
    ear_diff_max = df_landmarks['ear_diff'].max()
    
    # calculate the minimum difference in EAR
    ear_diff_min = df_landmarks['ear_diff'].min()
    
    # fetures dictionary
    blink_features_landmarks = {
        'blink_frequency_landmark': blink_frequency,
        'ear_diff_mean': ear_diff_mean,
        'ear_diff_variability': ear_diff_variability,
        'ear_diff_max': ear_diff_max,
        'ear_diff_min': ear_diff_min
    }
    
    return blink_features_landmarks
    

def calculate_static_features(distances, activity):
    """
    Calculate static features: mean, max, min, range, std deviation.

    Parameters:
    distances (list or np.array): Array of distances for a specific measure (e.g., lip parting).

    Returns:
    dict: Static features.
    """
    features = {
        activity+'_distance_mean': np.mean(distances),
        activity+'_distance_max': np.max(distances),
        activity+'_distance_min': np.min(distances),
        activity+'_distance_std': np.std(distances)
    }
    return features

# Calculate dynamic features
def calculate_dynamic_features(distances, fps=15, activity='lip_parting'):
    """
    Calculate dynamic features: speed, acceleration, jerk.
    
    Parameters:
    distances (list or np.array): Array of distances for a specific measure (e.g., lip parting).
    fps (int): Frame per second of the video.
    activity (str): Activity name (e.g., lip_parting).
    
    Returns:
    dict: Dynamic features.
    """
    # Calculate speed
    speeds = np.diff(distances) * fps
    
    # Calculate acceleration
    accelerations = np.diff(speeds) * fps
    
    # Calculate jerk
    jerks = np.diff(accelerations) * fps
    
    # Calculate mean, max, min, and std deviation of speed, acceleration, and jerk
    features = {
        activity+'_speed_mean': np.mean(speeds),
        activity+'_speed_max': np.max(speeds),
        activity+'_speed_min': np.min(speeds),
        activity+'_speed_std': np.std(speeds),
        activity+'_acceleration_mean': np.mean(accelerations),
        activity+'_acceleration_max': np.max(accelerations),
        activity+'_acceleration_min': np.min(accelerations),
        activity+'_acceleration_std': np.std(accelerations),
        activity+'_jerk_mean': np.mean(jerks),
        activity+'_jerk_max': np.max(jerks),
        activity+'_jerk_min': np.min(jerks),
        activity+'_jerk_std': np.std(jerks)
    }
    
    return features

def calculate_asymmetry_features(left_distances, right_distances, activity='lip_parting'):
    """
    Calculate symmetry features: mean, max, min, range, std deviation.

    Parameters:
    left_distances (list or np.array): Array of distances for the left side of the face.
    right_distances (list or np.array): Array of distances for the right side of the face.

    Returns:
    dict: Symmetry features.
    """
    # Calculate the difference between the left and right distances
    differences = np.abs(np.array(left_distances) - np.array(right_distances))
    
    # Calculate the mean, std deviation of the differences
    features = {
        activity+'_asymmetry_mean': np.mean(differences),
        activity+'_asymmetry_std': np.std(differences)
    }
    
    return features

# calculate tremor features 
def calculate_tremor_features(distances, activity, fps=15):
    """
    Calculate tremor frequency using FFT.

    Parameters:
    distances (list or np.array): Array of distances for a specific measure (e.g., lip parting).
    fps (int): Frame per second of the video.

    Returns:
    dict: Tremor frequency.
    """
    n = len(distances)
    freqs = np.fft.fftfreq(n, 1/fps)
    fft_values = np.fft.fft(distances)
    
    # only keep the positive frequencies
    freqs = freqs[0:n//2]
    fft_values = np.abs(fft_values[0:n//2])
    
    # find the tremor frequencies 
    # examine the frequency spectrum for significant peaks within the 3-12 Hz range for tremors
    tremor_indices = np.where((freqs >= 3) & (freqs <= 12))
    tremor_freqs = freqs[tremor_indices]
    tremor_values = fft_values[tremor_indices]
    
    if len(tremor_freqs) == 0:
        return {
            'max_tremor_freq': 0,
            'max_tremor_value': 0,
            'mean_tremor_freq': 0,
            'mean_tremor_value': 0,
            'total_tremor_power': 0
        }
    
    # calcaulte relavant features from these tremor frequencies and values
    max_freq_index = np.argmax(tremor_values)
    max_tremor_freq = tremor_freqs[max_freq_index]
    max_tremor_value = tremor_values[max_freq_index]
    mean_tremor_freq = np.mean(tremor_freqs)
    mean_tremor_value = np.mean(tremor_values)
    total_tremor_power = np.sum(tremor_values**2)
    total_signal_power = np.sum(fft_values**2)
    #normalized_tremor_power = total_tremor_power / total_signal_power
    normalized_tremor_power = safe_divide(total_tremor_power, total_signal_power)
    
    features = {
        activity+'_tremor_freq_max': max_tremor_freq,
        activity+'_tremor_value_max': max_tremor_value,
        activity+'_tremor_freq_mean': mean_tremor_freq,
        activity+'_tremor_value_mean': mean_tremor_value,
        activity+'_normalized_tremor_power': normalized_tremor_power
    }
    
    return features
    
    
    
def lips_puller_features_landmarks(df_landmarks):
    """
    Calculate lips puller features from MediaPipe landmarks.

    Parameters:
    df_landmarks (pd.DataFrame): DataFrame containing MediaPipe landmarks.

    Returns:
    dict: Lips puller features.
    """
    # calculate lips puller distance
    
    left_lip_landmark = '61'
    right_lip_landmark = '91'
    center_landmark = '0'
    
    df_landmarks['lips_puller_distance'] = calculate_pairwise_distances(df_landmarks, left_lip_landmark, right_lip_landmark)
    
    df_landmarks['left_lip_center_distance'] = calculate_pairwise_distances(df_landmarks, left_lip_landmark, center_landmark)

    df_landmarks['right_lip_center_distance'] = calculate_pairwise_distances(df_landmarks, right_lip_landmark, center_landmark)
        
    # calculate static features for lips puller distance
    lips_puller_features = calculate_static_features(df_landmarks['lips_puller_distance'], 'lips_puller')
    
    # calculate dynamic features for lips puller distance
    lips_puller_features.update(calculate_dynamic_features(df_landmarks['lips_puller_distance'], activity='lips_puller'))
    
    # calculate symmetry features for lips puller distance
    lips_puller_features.update(calculate_asymmetry_features(df_landmarks['left_lip_center_distance'], df_landmarks['right_lip_center_distance'], activity='lips_puller'))
    
    # calculate tremor features for lips puller distance
    lips_puller_features.update(calculate_tremor_features(df_landmarks['lips_puller_distance'], activity='lips_puller'))
    
    return lips_puller_features


# calculate static features for lip parting distance from landmarks
def lips_parting_features_landmkars(df_landmarks):
    """
    Calculate lip parting features from MediaPipe landmarks.

    Parameters:
    df_landmarks (pd.DataFrame): DataFrame containing MediaPipe landmarks.

    Returns:
    dict: Lip parting features.
    """
    # calculate lip parting distance
    
    upper_lip_landmark = '13'
    lower_lip_landmark = '14'
    
    df_landmarks['lips_parting_distance'] = calculate_pairwise_distances(df_landmarks, upper_lip_landmark, lower_lip_landmark)
    
    # calculate static features for lip parting distance
    lips_parting_features = calculate_static_features(df_landmarks['lips_parting_distance'], 'lips_parting')
    
    # calculate dynamic features for lip parting distance
    lips_parting_features.update(calculate_dynamic_features(df_landmarks['lips_parting_distance'], activity='lips_parting'))
    
    # calculate tremor features for lip parting distance
    lips_parting_features.update(calculate_tremor_features(df_landmarks['lips_parting_distance'], activity='lips_parting'))
    

    return lips_parting_features


def jaw_parting_features_landmarks(df_landmarks):
    """
    Calculate jaw parting features from MediaPipe landmarks.

    Parameters:
    df_landmarks (pd.DataFrame): DataFrame containing MediaPipe landmarks.

    Returns:
    dict: Jaw parting features.
    """
    # calculate jaw parting distance
    
    upper_jaw_landmark = '1'
    lower_jaw_landmark = '152'
    
    df_landmarks['jaw_parting_distance'] = calculate_pairwise_distances(df_landmarks, upper_jaw_landmark, lower_jaw_landmark)
    
    # calculate static features for jaw parting distance
    jaw_parting_features = calculate_static_features(df_landmarks['jaw_parting_distance'], 'jaw_parting')
    
    # calculate dynamic features for jaw parting distance
    jaw_parting_features.update(calculate_dynamic_features(df_landmarks['jaw_parting_distance'], activity='jaw_parting'))
    
    center_landmark = '0'
    left_jaw_landmark = '234'
    right_jaw_landmark = '454'
    
    df_landmarks['left_jaw_center_distance'] = calculate_pairwise_distances(df_landmarks, left_jaw_landmark, center_landmark)
    
    df_landmarks['right_jaw_center_distance'] = calculate_pairwise_distances(df_landmarks, right_jaw_landmark, center_landmark)
    
    # calculate symmetry features for jaw parting distance
    jaw_parting_features.update(calculate_asymmetry_features(df_landmarks['left_jaw_center_distance'], df_landmarks['right_jaw_center_distance'], activity='jaw_parting'))
    
    # calculate tremor features for jaw parting distance
    jaw_parting_features.update(calculate_tremor_features(df_landmarks['jaw_parting_distance'], activity='jaw_parting'))
    
    return jaw_parting_features


def calcaulte_other_symmetry_features(df_landmarks):
    """
    Inner Brow Raiser: 
        Left Brow Raiser = Dist(Left Inner Brow (223), Left Eye Center (33))
        Right Brow Raiser = Dist(Right Inner Brow (443), Right Eye Center (263))
        Symmetry = abs(Left Brow Raiser - Right Brow Raiser)
    Upper Lid Raiser
        Left Lid Distance = Distance(Left Upper Lid (386), Left Lower Lid (374))
        Right Lid Distance = Distance(Right Upper Lid (159), Right Lower Lid (145))
        Symmetry = abs(Left Lid Distance - Right Lid Distance)
    Cheek Raiser
        Left Cheek Distance = Distance(Left Cheek (276), Left Eye Center (33))
        Right Cheek Distance = Distance(Right Cheek (426), Right Eye Center (263))
        Symmetry = abs(Left Cheek Distance - Right Cheek Distance)

    """ 
    
    left_eye_center_landmark = '33'
    left_inner_brow_landmark = '223'
    right_eye_center_landmark = '263'
    right_inner_brow_landmark = '443'
    
    df_landmarks['left_brow_raiser_distance'] = calculate_pairwise_distances(df_landmarks, left_inner_brow_landmark, left_eye_center_landmark)
    
    df_landmarks['right_brow_raiser_distance'] = calculate_pairwise_distances(df_landmarks, right_inner_brow_landmark, right_eye_center_landmark)

    
    # calculate symmetry features for inner brow raiser
    inner_brow_raiser_features = calculate_asymmetry_features(df_landmarks['left_brow_raiser_distance'], df_landmarks['right_brow_raiser_distance'], activity='inner_brow_raiser')
    
    left_upper_lid_landmark = '386'
    left_lower_lid_landmark = '374'
    right_upper_lid_landmark = '159'
    right_lower_lid_landmark = '145'
    
    df_landmarks['left_lid_distance'] = calculate_pairwise_distances(df_landmarks, left_upper_lid_landmark, left_lower_lid_landmark)
    df_landmarks['right_lid_distance'] = calculate_pairwise_distances(df_landmarks, right_upper_lid_landmark, right_lower_lid_landmark)
    
    # calculate symmetry features for upper lid raiser
    upper_lid_raiser_features = calculate_asymmetry_features(df_landmarks['left_lid_distance'], df_landmarks['right_lid_distance'], activity='upper_lid_raiser')
    
    left_cheek_landmark = '276'
    right_cheek_landmark = '426'
    
    df_landmarks['left_cheek_distance'] = calculate_pairwise_distances(df_landmarks, left_cheek_landmark, left_eye_center_landmark)
    df_landmarks['right_cheek_distance'] = calculate_pairwise_distances(df_landmarks, right_cheek_landmark, right_eye_center_landmark)
    
    # calculate symmetry features for cheek raiser
    cheek_raiser_features = calculate_asymmetry_features(df_landmarks['left_cheek_distance'], df_landmarks['right_cheek_distance'], activity='cheek_raiser')
    
    features = {}
    features.update(inner_brow_raiser_features)
    features.update(upper_lid_raiser_features)
    features.update(cheek_raiser_features)
    
    return features

# we will calculate the vector angle between v1 and v2 
# v1 = Forehead Center → Nose Bridge
# v2 = Nose Bridge → Chin
def calculate_vector_angle(v1, v2):
    """
    Calculate the angle between two vectors.

    Parameters:
    v1 (np.array): Vector 1.
    v2 (np.array): Vector 2.

    Returns:
    float: Angle between two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    magnitude = norm_v1 * norm_v2
    if magnitude == 0:
        return 0
    # cos_theta = dot_product / magnitude
    cos_theta = safe_divide(dot_product, magnitude)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

# features to capture relative movements of different facial landmarks
def calculate_relative_movement_features(df_landmarks):
    """
    Calculate relative movement features from MediaPipe landmarks.

    Parameters:
    df_landmarks (pd.DataFrame): DataFrame containing MediaPipe landmarks.

    Returns:
    dict: Relative movement features.
    """
    # calculate relative movement features
    forehead_center_landmark = '10'
    nose_bridge_landmark = '6'
    chin_landmark = '152'
    left_cheek_landmark = '234'
    right_cheek_landmark = '454'
    
    df_landmarks['forehead_nose_distance'] = calculate_pairwise_distances(df_landmarks, forehead_center_landmark, nose_bridge_landmark)
    df_landmarks['chin_nose_distance'] = calculate_pairwise_distances(df_landmarks, chin_landmark, nose_bridge_landmark)
    df_landmarks['cheek_distance'] = calculate_pairwise_distances(df_landmarks, left_cheek_landmark, right_cheek_landmark)
    
    # calculate static features for relative movement features
    relative_movement_features = calculate_static_features(df_landmarks['forehead_nose_distance'], 'forehead_nose')
    relative_movement_features.update(calculate_static_features(df_landmarks['chin_nose_distance'], 'chin_nose'))
    relative_movement_features.update(calculate_static_features(df_landmarks['cheek_distance'], 'cheek'))
    
    # calculate dynamic features for relative movement features
    relative_movement_features.update(calculate_dynamic_features(df_landmarks['forehead_nose_distance'], activity='forehead_nose'))
    relative_movement_features.update(calculate_dynamic_features(df_landmarks['chin_nose_distance'], activity='chin_nose'))
    relative_movement_features.update(calculate_dynamic_features(df_landmarks['cheek_distance'], activity='cheek'))
    
    # calculate the angle between the vectors
    angles = []
    
    angles_yaw = []
    
    for i in range(len(df_landmarks)):
        forehead = np.array([df_landmarks[f'{forehead_center_landmark}_x'].iloc[i], df_landmarks[f'{forehead_center_landmark}_y'].iloc[i]])
        nose_bridge = np.array([df_landmarks[f'{nose_bridge_landmark}_x'].iloc[i], df_landmarks[f'{nose_bridge_landmark}_y'].iloc[i]])
        chin = np.array([df_landmarks[f'{chin_landmark}_x'].iloc[i], df_landmarks[f'{chin_landmark}_y'].iloc[i]])
        
        v1 = forehead - nose_bridge
        v2 = chin - nose_bridge
        
        angles.append(calculate_vector_angle(v1, v2))
        
        # angle_yaw = np.arctan((df_landmarks[f'{right_cheek_landmark}_y'].iloc[i] - df_landmarks[f'{left_cheek_landmark}_y'].iloc[i]) / (df_landmarks[f'{right_cheek_landmark}_x'].iloc[i] - df_landmarks[f'{left_cheek_landmark}_x'].iloc[i]))
        angle_yaw = np.arctan(
            safe_divide(
                (df_landmarks[f'{right_cheek_landmark}_y'].iloc[i] - df_landmarks[f'{left_cheek_landmark}_y'].iloc[i]),
                (df_landmarks[f'{right_cheek_landmark}_x'].iloc[i] - df_landmarks[f'{left_cheek_landmark}_x'].iloc[i])
            )
        )
        angles_yaw.append(angle_yaw)
    
    # calculate static and dynamic features for angle
    relative_movement_features.update(calculate_static_features(angles, activity='angle'))
    relative_movement_features.update(calculate_dynamic_features(angles, activity='angle'))
    relative_movement_features.update(calculate_tremor_features(angles, activity='angle'))
    relative_movement_features.update(calculate_static_features(angles_yaw, activity='angle_yaw'))
    relative_movement_features.update(calculate_dynamic_features(angles_yaw, activity='angle_yaw'))
    relative_movement_features.update(calculate_tremor_features(angles_yaw, activity='angle_yaw'))
    
    return relative_movement_features



no_landmark_dict = {}

minimum_frame_detected_for_landmarks = fps * 2

def extract_mediapipe_landmarks(video_file_path):
    # create a dataframe that will contain the mediapipe features
    columns = ['frame_number']
    for i in range(478):
        columns.append(str(i)+'_x')
        columns.append(str(i)+'_y')
        columns.append(str(i)+'_z')
    columns.append('reference_distance')
    mediapipe_landmark_df = pd.DataFrame(columns=columns)
    
    videocap = cv2.VideoCapture(video_file_path)
    # read the first frame
    success, image = videocap.read()
    # initialize the frame count
    count = 0
    # iterate through all the frames in the video
    while success:
        # create face mesh object
        face_mesh = mp.solutions.face_mesh.FaceMesh(
                            static_image_mode=True,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                        )
        # convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # process the image
        results = face_mesh.process(image_rgb)
        # if landmarks are detected
        if results.multi_face_landmarks:
            # convert the landmarks to a dictionary
            landmarks_dict = protobuf_to_dict(results.multi_face_landmarks[0])
            # create a list to store the landmarks
            landmark_list = [count]
            for landmark in landmarks_dict['landmark']:
                landmark_list.append(landmark['x'])
                landmark_list.append(landmark['y'])
                landmark_list.append(landmark['z'])
            # calculate the reference distance
            reference_distance = np.sqrt((landmarks_dict['landmark'][int(landmark_reference_points['reference_point1'])]['x'] - landmarks_dict['landmark'][int(landmark_reference_points['reference_point2'])]['x'])**2 + (landmarks_dict['landmark'][int(landmark_reference_points['reference_point1'])]['y'] - landmarks_dict['landmark'][int(landmark_reference_points['reference_point2'])]['y'])**2 + (landmarks_dict['landmark'][int(landmark_reference_points['reference_point1'])]['z'] - landmarks_dict['landmark'][int(landmark_reference_points['reference_point2'])]['z'])**2)
            landmark_list.append(reference_distance)
            # add the landmarks to the dataframe
            mediapipe_landmark_df.loc[len(mediapipe_landmark_df)] = landmark_list
        else:
            if video_file_path in no_landmark_dict:
                no_landmark_dict[video_file_path].append(count)
                # print(f"No landmarks detected for frame {count} of video {video_file_path}")
            else:
                no_landmark_dict[video_file_path] = [count]
                # print(f"No landmarks detected for frame {count} of video {video_file_path}")
        # release the face mesh object
        face_mesh.close()
        # read the next frame
        success, image = videocap.read()
        # increment the frame count
        count += 1
    # release the video capture object
    videocap.release()
    return mediapipe_landmark_df



@click.command()
@click.option("--subdirectory",default='finger_tapping')
def main(**args):
    
    failed_videos = []
    
    root_directory = "/Users/tariqadnan/Research/Co Learning/standardized"

    # the first user argument is the subdirectory
    
    subdirectory = args['subdirectory']
    subdirectory_path = os.path.join(root_directory, subdirectory)

    full_features_df = pd.DataFrame()
    print("Processing subdirectory: ", subdirectory_path)
    if os.path.isdir(subdirectory_path):
        list_of_files = os.listdir(subdirectory_path)
        # print("list of files in the subdirectory: ", list_of_files)
        print("Number of files in the subdirectory: ", len(list_of_files))
        for i in tqdm(range(len(list_of_files)), file=sys.stdout, desc="Processing files"):
            file = list_of_files[i]
            if file.endswith('.mp4'):
                file_path = os.path.join(subdirectory_path, file)
                df_landmarks = extract_mediapipe_landmarks(file_path)
                # try:
                #     df_landmarks = pd.read_csv(os.path.join(subdirectory_path, file[:-3] + '-mediapipe-landmarks.csv'))
                # except:
                #     continue
                if len(df_landmarks) > minimum_frame_detected_for_landmarks:
                    # df_landmarks.to_csv(os.path.join(subdirectory_path, file[:-3] + '-mediapipe-landmarks.csv'), index=False)
                    features = {
                        'filename': file[:-3]
                    }
                    features.update(blink_features_landmaks(df_landmarks))
                    features.update(lips_puller_features_landmarks(df_landmarks))
                    features.update(lips_parting_features_landmkars(df_landmarks))
                    features.update(jaw_parting_features_landmarks(df_landmarks))
                    features.update(calcaulte_other_symmetry_features(df_landmarks))
                    features.update(calculate_relative_movement_features(df_landmarks))
                    features_df = pd.DataFrame([features])
                    features_df.to_csv(os.path.join(subdirectory_path, file[:-3] + '-mediapipe-features.csv'), index=False)
                    full_features_df = pd.concat([full_features_df, features_df], ignore_index=True)
                else:
                    #print(f"No landmarks detected for video {file_path}") --> make it to print to stderr
                    failed_videos.append(file_path)
                
                    

    # save the list of failed videos
    with open(args['subdirectory']+'_failed_videos.txt', 'w') as f:
        for video in failed_videos:
            f.write(video + '\n')

    full_features_df.to_csv(args['subdirectory']+"-mediapipe_features.csv", index=False)

    # finally dump the no_landmark_dict to a file
    import json

    with open(args['subdirectory']+'_no_landmark_dict.json', 'w') as f:
        json.dump(no_landmark_dict, f)

    
if __name__ == "__main__":
    print("Starting the process")
    main()
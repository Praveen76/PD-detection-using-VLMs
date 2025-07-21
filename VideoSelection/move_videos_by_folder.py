import os
import shutil

# read remaining annotations
filename = "/localdisk1/PARK/park_vlm/remaining_annotations.txt"

with open(filename, "r") as f:
    # create a folder for nami
    nami_folder = "/localdisk1/PARK/park_vlm/VideoSelection/Nami-R3"
    if not os.path.exists(nami_folder):
        os.makedirs(nami_folder)

    # skip one line
    line1 = f.readline()
    while True:
        line = f.readline()
        video_filename = line.strip()
        if video_filename=="":
            break
        
        source = os.path.join("/localdisk1/PARK/park_vlm/Videos", video_filename)
        dest = os.path.join(nami_folder, video_filename)
        # shutil.copy2(src=source, dst=dest)

    # create a folder for natalia
    natalia_folder = "/localdisk1/PARK/park_vlm/VideoSelection/Natalia-R3"
    if not os.path.exists(natalia_folder):
        os.makedirs(natalia_folder)

    # skip two lines
    f.readline()
    f.readline()
    while True:
        line = f.readline()
        
        video_filename = line.strip()
        if video_filename=="":
            break
        
        source = os.path.join("/localdisk1/PARK/park_vlm/Videos", video_filename)
        dest = os.path.join(natalia_folder, video_filename)
        # shutil.copy2(src=source, dst=dest)

    # create a folder for Cayla-common
    cayla_common_folder = "/localdisk1/PARK/park_vlm/VideoSelection/Cayla-R1"
    if not os.path.exists(cayla_common_folder):
        os.makedirs(cayla_common_folder)

    # skip two lines
    f.readline()
    f.readline()
    while True:
        line = f.readline()
        
        video_filename = line.strip()
        if video_filename=="":
            break
        
        source = os.path.join("/localdisk1/PARK/park_vlm/Videos", video_filename)
        dest = os.path.join(cayla_common_folder, video_filename)
        # shutil.copy2(src=source, dst=dest)

    # create a folder for Cayla-unique
    cayla_unique_folder = "/localdisk1/PARK/park_vlm/VideoSelection/Cayla-R2"
    if not os.path.exists(cayla_unique_folder):
        os.makedirs(cayla_unique_folder)

    # skip two lines
    f.readline()
    f.readline()
    while True:
        line = f.readline()
        if line is None:
            break
        
        video_filename = line.strip()
        if video_filename=="":
            break
        
        source = os.path.join("/localdisk1/PARK/park_vlm/Videos", video_filename)
        dest = os.path.join(cayla_unique_folder, video_filename)
        # shutil.copy2(src=source, dst=dest)



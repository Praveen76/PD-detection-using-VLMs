import urllib.request
import ssl
import os
from os import system
import socket
import sys
import time
current_time_ms = round(time.time() * 1000)

old_stdout = sys.stdout

log_file = open(f"/localdisk1/PARK/park_vlm/video_downloader_log_{current_time_ms}.txt","w")
sys.stdout = log_file

socket.setdefaulttimeout(30)

ssl._create_default_https_context = ssl._create_unverified_context

video_url_list_dev = [
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_mp4_partaa__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoOeFjwyo$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_mp4_partab__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoqY0mTZE$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_mp4_partac__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoyA6mUpo$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_mp4_partad__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoo4RggTI$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_mp4_partae__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbho4bna6TQ$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_mp4_partaf__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoDmBC2VM$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_mp4_partag__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoFYJIbcI$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_mp4_partah__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoEbOZhIU$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_mp4_partai__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhozaDOJWQ$"
    ]

video_filenames_dev = [f"vox2_dev_mp4_parta{chr(ord('a')+i)}" for i in range(len(video_url_list_dev))]

#   download each video file individually
indexes = []
current_index = 0
for url, filename in zip(video_url_list_dev, video_filenames_dev):
    if current_index in indexes:
        print(f"Downloading file {filename} ...")
        sys.stdout.flush()
        retries = 20
        attempt = 1
        while attempt<=retries:
            try:
                print(f"Attempt {attempt}")
                sys.stdout.flush()
                urllib.request.urlretrieve(url, os.path.join("/localdisk1/voxceleb/raw_downloaded", filename))
                print(f"File '{filename}' downloaded successfully.")
                sys.stdout.flush()
                break
            except Exception as e:
                print(f"Error {e}")
                sys.stdout.flush()
                attempt +=1

    current_index +=1
    
sys.stdout.flush()
sys.stdout = old_stdout
log_file.close()

#   concatenate the video files into a huge zip file
system("cat /localdisk1/voxceleb/raw_downloaded/vox2_dev_mp4* > /localdisk1/voxceleb/raw_downloaded/vox2_mp4.zip")
print("All training data combined into a single zip file")

trial = 1
while True:
    #   download the test files
    video_test_url = "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_test_mp4.zip__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoxM9Lz1s$"
    video_test_filename = "vox2_test_mp4.zip"
    print(f"Downloading file {video_test_filename} ...")
    try:
        print(f"Attempt {trial}")
        urllib.request.urlretrieve(video_test_url, os.path.join("/localdisk1/voxceleb/raw_downloaded", video_test_filename))
        print(f"File '{video_test_filename}' downloaded successfully.")
        break
    except Exception as e:
        print(f"Error {e}")
    trial +=1

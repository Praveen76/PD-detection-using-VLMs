import urllib.request
import ssl
import os
from os import system

ssl._create_default_https_context = ssl._create_unverified_context

audio_url_list_dev = [
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_aac_partaa__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhomKp-FyI$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_aac_partab__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhopeKbavU$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_aac_partac__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhovM1N5Co$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_aac_partad__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoQvfF4Mw$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_aac_partae__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoUYiSvAI$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_aac_partaf__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbho61UQ88E$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_aac_partag__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoQ6oP9XA$",
    "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_dev_aac_partah__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoWNje-kU$"
    ]

audio_filenames_dev = [f"vox2_dev_aac_parta{chr(ord('a')+i)}" for i in range(len(audio_url_list_dev))]

#   download each audio file individually
start_index = 8
end_index = 8
current_index = 0
for url, filename in zip(audio_url_list_dev, audio_filenames_dev):
    if current_index>=start_index and current_index<=end_index:
        print(f"Downloading file {filename} ...")
        urllib.request.urlretrieve(url, os.path.join("/localdisk1/voxceleb/raw_downloaded", filename))
        print(f"File '{filename}' downloaded successfully.")
    current_index +=1

#   concatenate the audio files into a huge zip file
system("cat /localdisk1/voxceleb/raw_downloaded/vox2_dev_aac* > /localdisk1/voxceleb/raw_downloaded/vox2_aac.zip")

#   download the test files
audio_test_url = "https://urldefense.com/v3/__https://cn01.mmai.io/download/voxceleb?key=19df48c312d2390eb363ddabf3d160368d1e37e00a1119324c182613160c6d7874c71fcd314a78b79d2abe959da19353b03b735ac43652b6f84f58575c881bc7ac3240412193acc10494a7eb9f41b0621d8f6830d24b6cf375e1fc9a52271bbac3eec7f5e6205175c64a201f47c756c4c335d0559e590f5d36bce8184235235b&file=vox2_test_aac.zip__;!!CGUSO5OYRnA7CQ!cB0fzD7WNFiwkIM_xRukgQwXtWDn7wyLt2O0dccxfgYWcqXZfmG9ivfDbtDy59lEhk0S9N1XXcw0sUzzGbhoziLyzWE$"
audio_test_filename = "vox2_test_aac.zip"
print(f"Downloading file {audio_test_filename} ...")
urllib.request.urlretrieve(audio_test_url, os.path.join("/localdisk1/voxceleb/raw_downloaded", audio_test_filename))
print(f"File '{audio_test_filename}' downloaded successfully.")

#   finally unzip them to extract the audios
#   unzip vox2_test_aac.zip
#   unzip vox2_aac.zip

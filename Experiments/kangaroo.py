
import json
import os
import boto3
import yaml
from google.colab import drive

# Mount Google Drive to access API keys
drive.mount('/content/drive', force_remount=True)

# Load API keys from file
file_path = '/content/drive/MyDrive/.API_KEYS/API_KEYS.yml'

with open(file_path, 'r') as file:
    api_keys = yaml.safe_load(file)

### WANDB Keys
wandb_key = api_keys['WANDB']['Key']
hf_read_api_key = api_keys['HUGGINGFACE']['HF_READ_API_KEY']

# Extract AWS credentials
aws_access_key_id = api_keys['AWS']['AWS_ACCESS_KEY_ID']
aws_secret_access_key = api_keys['AWS']['AWS_SECRET_ACCESS_KEY']

from huggingface_hub import login
login(hf_read_api_key)

# Initialize the S3 client with credentials
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

import wandb
wandb.login(key=wandb_key)


from huggingface_hub import login
login(hf_read_api_key)

import os
print(os.environ.get("HF_TOKEN"))
print(hf_read_api_key)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("KangarooGroup/kangaroo")
model = AutoModelForCausalLM.from_pretrained(
    "KangarooGroup/kangaroo",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
# model = model.to("cuda")
model = model


run = wandb.init(project="kangaroo_zero_shot", config={
    "model": "KangarooGroup/kangaroo",
    "task": "Zero-shot prompting on video",
    "max_new_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.9
})

# Step 4: Load the Kangaroo model and tokenizer using Hugging Face Transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("KangarooGroup/kangaroo")
model = AutoModelForCausalLM.from_pretrained(
    "KangarooGroup/kangaroo",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    # token=hf_read_api_key
)
# Uncomment the following line if using a GPU
# model = model.to("cuda")
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# Step 5: Define video input path and the zero-shot prompt query
video_path = "/content/data/pickleFiles/processed_File1.pkl"
query = '''
Imagine you are a clinician specializing in movement disorders. Rely on your knowledge of neurology and clinical care. Now, you are watching a home-recorded video of a person performing some tasks used to assess Parkinson's disease. No experts supervise the person, so there can be different types of noise, or the person may not follow the task instructions properly. The person can also show symptoms that may be associated with having Parkinson's disease. Focus on the noises, task instructions, user compliance, and possible symptoms of Parkinson's disease while answering the question.

Task instructions: The person will talk about a recent book they have read or a movie or TV show they have watched. The person will speak for approximately one minute. They should be front-facing the camera, and their face must be visible in the recording frame. There should not be any other person visible in the recording frame. The background should not be dark or overlit and should have good contrast against the person's face. For this task, the face is the most crucial body part you should focus on. However, you should also observe other body parts for relevant symptoms or signs of Parkinson's disease.

Answer the question about what is happening in the video.

Question: Please describe whether the person demonstrates any difficulty through their facial expressions. Some examples of visible difficulty include furrowed brow, squinting eyes, clenched jaw, tight lips, head hanging low, sighing, wrinkled forehead, etc. Mention such specific details when found. End output with a final answer choice: “Yes” or “No”.

Answer:
'''

# Step 6: Run the model's chat function and log the output to WANDB
out, history = model.chat(
    video_path=video_path,
    query=query,
    tokenizer=tokenizer,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    use_cache=False,
)
print("Assistant:\n", out)
wandb.log({"response": out})

# Step 7: Finish the wandb run
run.finish()

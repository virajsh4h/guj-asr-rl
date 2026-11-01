#!/bin/bash

echo "=== 1. Updating system and installing Node.js / FFmpeg ==="
sudo apt-get update -y && sudo apt-get install -y nodejs npm ffmpeg

echo "=== 2. Installing Localtunnel globally ==="
sudo npm install -g localtunnel

echo "=== 3. Installing Python ML packages ==="
pip install datasets[audio] renumics-spotlight soundfile

echo "=== 4. Creating the Hugging Face Stream script ==="
cat << 'PYEOF' > app.py
from datasets import load_dataset
from renumics import spotlight
import os

print("Streaming Gujarati subset validation split...")
ds = load_dataset(
    "ai4bharat/IndicVoices", 
    name="gujarati", 
    split="valid", 
    streaming=True
)

# Pull 100 rows into a list so memory remains highly optimized
sub_ds = list(ds.take(100))

print("Launching Spotlight GUI locally...")
# We use wait=True so the Python loop keeps running in the background script
spotlight.show(
    sub_ds, 
    host="127.0.0.1", 
    port=8501,
    wait=True
)
PYEOF

echo "=== 5. Starting the Public URL Tunnel ==="
# Launches localtunnel in the background pointing to your app port
lt --port 8501 > localtunnel.log 2>&1 &

echo "=== 6. Launching the App ==="
echo "--------------------------------------------------------"
echo "Initializing server. Your temporary public link will show up below..."
echo "--------------------------------------------------------"
sleep 3
# Print out the assigned URL from the tunnel logs
cat localtunnel.log

# Run the python app to process the dataset audio files
python3 app.py

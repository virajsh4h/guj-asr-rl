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

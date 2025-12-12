#!/usr/bin/env python3
"""
Download and process AIME dataset from JSONL files.
Mimics the structure of geo3k/process_data.py
"""

import os
import json
from datasets import Dataset
from PIL import Image
from io import BytesIO
import base64
from pathlib import Path


def normalize_image_for_source(image):
    """
    Normalize image to PIL.Image.Image or string path format.
    Adapted from mmmu_download.ipynb
    """
    # 1) Already PIL Image
    if isinstance(image, Image.Image):
        return image

    # 2) HuggingFace Image: dict with 'bytes'/'path'
    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            bio = BytesIO(image["bytes"])
            bio.seek(0)
            return Image.open(bio).convert("RGB")
        if "path" in image and image["path"]:
            return image["path"]

    # 3) Direct bytes
    if isinstance(image, (bytes, bytearray)):
        bio = BytesIO(image)
        bio.seek(0)
        return Image.open(bio).convert("RGB")

    # 4) BytesIO
    if isinstance(image, BytesIO):
        image.seek(0)
        return Image.open(image).convert("RGB")

    # 5) String: http(s) / file:// / data:image / local path
    if isinstance(image, str):
        if image.startswith("data:image") and "base64," in image:
            b64 = image.split("base64,", 1)[1]
            data = base64.b64decode(b64)
            bio = BytesIO(data)
            bio.seek(0)
            return Image.open(bio).convert("RGB")
        return image

    # 6) Try PIL directly
    try:
        return Image.open(image).convert("RGB")
    except Exception:
        raise TypeError(f"Unsupported image type: {type(image)}")


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    data = []
    print(f"Loading dataset from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples")
    return data


def make_map_fn(split):
    def process_fn(example, idx):
        # Extract fields based on actual dataset structure
        problem = example.get("question", example.get("problem", example.get("text", "")))
        answer = example.get("answer", example.get("solution", example.get("ground_truth", "")))
        
        # Build prompt
        prompt = problem.strip()
            
        
        # Build messages format
        messages = [{
            "role": "user",
            "content": prompt.strip(),
        }]
        
        data = {
            "data_source": "aime",
            "prompt": messages,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": problem,
            },
        }
        return data
    return process_fn


# Process each split
base_dir = Path(__file__).parent
local_dir = base_dir 
os.makedirs(local_dir, exist_ok=True)

# Define file mappings
file_mappings = [
    ("AIME_Dataset_1983-2023.jsonl", "train"),
    ("AIME_Dataset_2024-2025.jsonl", "test"),
]

for jsonl_file, split_name in file_mappings:
    jsonl_path = base_dir / jsonl_file
    
    if not jsonl_path.exists():
        print(f"Skipping {split_name} (file not found: {jsonl_path})")
        continue
    
    print(f"\nProcessing {split_name} split...")
    
    # Load JSONL data
    data = load_jsonl(jsonl_path)
    
    if len(data) == 0:
        print(f"Skipping {split_name} (empty)")
        continue
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    
    # Process the dataset
    process_fn = make_map_fn(split_name)
    processed_dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,  # Adjust based on your system
        desc=f"Processing {split_name}"
    )
    
    # Save to parquet
    output_file = local_dir / f"aime_{split_name}.parquet"
    processed_dataset.to_parquet(output_file)
    print(f"Saved {len(processed_dataset)} samples to {output_file}")

print("\nAll splits processed successfully!")


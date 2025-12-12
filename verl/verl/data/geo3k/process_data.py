#!/usr/bin/env python3
"""
Download and process geometry3k dataset from HuggingFace.
Mimics the structure of mmmu_download.ipynb
"""

import os
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import base64


dataset_name = "hiyouga/geometry3k"

print(f"Loading dataset: {dataset_name}")

# Load the dataset
dataset = load_dataset(dataset_name)

print(f"Dataset splits: {list(dataset.keys())}")

# Check the structure of the dataset
print("\n" + "="*50)
print("Dataset Structure:")
print("="*50)
for split_name in dataset.keys():
    split_data = dataset[split_name]
    print(f"\n{split_name} split: {len(split_data)} samples")
    if len(split_data) > 0:
        sample = split_data[0]
        print(f"  Column names: {list(sample.keys())}")
        print(f"  Sample data structure:")
        for key, value in sample.items():
            if key == "images":
                print(f"    {key}: {type(value).__name__} (length: {len(value) if isinstance(value, list) else 'N/A'})")
            else:
                value_str = str(value)
                if len(value_str) > 100:
                    print(f"    {key}: {value_str}...")
                else:
                    print(f"    {key}: {value}")
print("="*50 + "\n")


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


def make_map_fn(split):
    def process_fn(example, idx):
        # Extract fields based on actual dataset structure
        # Adjust these based on the actual column names in the dataset
        problem = example.get("question", example.get("problem", example.get("text", "")))
        answer = example.get("answer", example.get("solution", example.get("ground_truth", "")))
        
        # Build prompt
        prompt = problem.strip()
        
        # Handle images - images should be a list
        images = []
        # Check if 'images' field exists (it should be a list)
        if "images" in example and example["images"] is not None:
            images_raw = example["images"]
            # Ensure it's a list
            if not isinstance(images_raw, list):
                images_raw = [images_raw]
            
            # Process each image in the list
            for img in images_raw:
                if img is not None:
                    try:
                        normalized = normalize_image_for_source(img)
                        images.append(normalized)
                    except Exception as e:
                        print(f"Warning: Failed to process image at index {idx}: {e}")
        
        # Build messages format
        messages = [{
            "role": "user",
            "content": prompt.strip(),
        }]
        
        data = {
            "data_source": "hiyouga/geometry3k",
            "prompt": messages,
            "images": images,  # Always a list, even if empty
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
local_dir = "./datasets"
os.makedirs(local_dir, exist_ok=True)

for split_name in dataset.keys():
    print(f"\nProcessing {split_name} split...")
    split_dataset = dataset[split_name]
    
    if len(split_dataset) == 0:
        print(f"Skipping {split_name} (empty)")
        continue
    
    # Process the dataset
    process_fn = make_map_fn(split_name)
    processed_dataset = split_dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,  # Adjust based on your system
        desc=f"Processing {split_name}"
    )
    
    # Save to parquet
    output_file = os.path.join(local_dir, f"geometry3k_{split_name}.parquet")
    processed_dataset.to_parquet(output_file)
    print(f"Saved {len(processed_dataset)} samples to {output_file}")

print("\nAll splits processed successfully!")


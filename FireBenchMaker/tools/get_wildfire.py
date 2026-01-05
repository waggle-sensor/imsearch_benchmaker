#!/usr/bin/env python3
"""
get_wildfire.py

Download a sample of images from The Wildfire Dataset on Kaggle for the FireBench benchmark.
Samples 650 fire and 650 nofire images from the train split.
"""
import os
import shutil
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from kaggle.api.kaggle_api_extended import KaggleApi
from config import RANDOM_SEED

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATASET = "elmadafri/the-wildfire-dataset"
TRAIN_SPLIT = "the_wildfire_dataset_2n_version/train"
FIRE_CLASS = "fire"
NOFIRE_CLASS = "nofire"
FIRE_SAMPLE_SIZE = 650
NOFIRE_SAMPLE_SIZE = 650
MAX_WORKERS = 10
OUTPUT_DIR = "The-Wildfire-Dataset"

def copy_image(src_path, dst_path):
    """Copy a single image file."""
    try:
        if os.path.exists(dst_path):
            return {"status": "skipped", "filepath": str(dst_path)}
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return {"status": "success", "filepath": str(dst_path)}
    except Exception as e:
        logger.error(f"Failed to copy {src_path}: {e}")
        return {"status": "error", "filepath": str(dst_path), "error": str(e)}

def get_image_files(directory, class_name):
    """Get all image files from a class directory."""
    class_dir = Path(directory) / class_name
    if not class_dir.exists():
        logger.warning(f"Directory {class_dir} does not exist")
        return []
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in class_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    return image_files

def sample_and_copy_images(source_dir, output_dir, fire_count, nofire_count, random_state=RANDOM_SEED):
    """Sample and copy images from train split."""
    random.seed(random_state)
    logger.info(f"Using random seed: {random_state}")
    
    # Get all image files
    logger.info("Collecting fire images...")
    fire_images = get_image_files(source_dir, FIRE_CLASS)
    logger.info(f"Found {len(fire_images)} fire images")
    
    logger.info("Collecting nofire images...")
    nofire_images = get_image_files(source_dir, NOFIRE_CLASS)
    logger.info(f"Found {len(nofire_images)} nofire images")
    
    # Sample images
    if len(fire_images) < fire_count:
        logger.warning(f"Only {len(fire_images)} fire images available, using all")
        sampled_fire = fire_images
    else:
        sampled_fire = random.sample(fire_images, fire_count)
        logger.info(f"Sampled {fire_count} fire images from {len(fire_images)} total")
    
    if len(nofire_images) < nofire_count:
        logger.warning(f"Only {len(nofire_images)} nofire images available, using all")
        sampled_nofire = nofire_images
    else:
        sampled_nofire = random.sample(nofire_images, nofire_count)
        logger.info(f"Sampled {nofire_count} nofire images from {len(nofire_images)} total")
    
    # Prepare copy tasks
    tasks = []
    for img_path in sampled_fire:
        # Preserve relative path structure
        rel_path = img_path.relative_to(source_dir)
        dst_path = Path(output_dir) / rel_path
        tasks.append((img_path, dst_path))
    
    for img_path in sampled_nofire:
        rel_path = img_path.relative_to(source_dir)
        dst_path = Path(output_dir) / rel_path
        tasks.append((img_path, dst_path))
    
    logger.info(f"Starting copy of {len(tasks)} images to {output_dir}")
    
    # Copy in parallel
    results = {"success": 0, "skipped": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(copy_image, src, dst)
            for src, dst in tasks
        ]
        
        for future in as_completed(futures):
            result = future.result()
            results[result["status"]] += 1
    
    logger.info(f"Completed: {results['success']} success, "
                f"{results['skipped']} skipped, {results['error']} errors")
    return results

def download_dataset(dataset_name, extract_dir="kaggle_temp"):
    """Download dataset from Kaggle."""
    api = KaggleApi()
    api.authenticate()
    
    logger.info(f"Downloading dataset {dataset_name}...")
    api.dataset_download_files(
        dataset_name,
        path=extract_dir,
        unzip=True,
        quiet=False
    )
    
    # Find the train directory
    extract_path = Path(extract_dir)
    train_dir = None
    
    # Look for train directory in various possible locations
    possible_paths = [
        extract_path / TRAIN_SPLIT,
        extract_path / "train",
        extract_path / "the_wildfire_dataset_2n_version" / "train",
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            train_dir = path
            break
    
    if train_dir is None:
        # Try to find it recursively
        for path in extract_path.rglob("train"):
            if path.is_dir() and (path / FIRE_CLASS).exists() and (path / NOFIRE_CLASS).exists():
                train_dir = path
                break
    
    if train_dir is None:
        raise FileNotFoundError(
            f"Could not find train directory. Searched in: {[str(p) for p in possible_paths]}"
        )
    
    logger.info(f"Found train directory at: {train_dir}")
    return train_dir

def main():
    """Main function to download and sample Wildfire dataset images."""
    # Check if dataset is already downloaded
    temp_dir = "kaggle_temp"
    train_dir = None
    
    # Check if temp directory exists and has train data
    if os.path.exists(temp_dir):
        possible_train = Path(temp_dir) / TRAIN_SPLIT
        if possible_train.exists():
            train_dir = possible_train
            logger.info(f"Using existing dataset at {train_dir}")
    
    # Download if needed
    if train_dir is None:
        try:
            train_dir = download_dataset(DATASET, temp_dir)
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.error("Make sure you have Kaggle API credentials set up:")
            logger.error("1. Go to https://www.kaggle.com/settings and create an API token")
            logger.error("2. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars")
            return
    
    # Sample and copy images
    try:
        sample_and_copy_images(
            train_dir,
            OUTPUT_DIR,
            FIRE_SAMPLE_SIZE,
            NOFIRE_SAMPLE_SIZE,
            random_state=RANDOM_SEED
        )
        logger.info("All downloads completed!")
        
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                logger.info("Temporary directory deleted successfully")
            except Exception as e:
                logger.warning(f"Failed to delete temporary directory: {e}")
    except Exception as e:
        logger.error(f"Failed to sample images: {e}")
        raise

if __name__ == "__main__":
    main()


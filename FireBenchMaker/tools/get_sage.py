#!/usr/bin/env python3
"""
get_sage.py

Download images from Sage that will be used for the FireBench benchmark.
"""
import sage_data_client
import requests
from PIL import Image
from io import BytesIO
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from urllib.parse import urlparse
import pandas as pd
from config import RANDOM_SEED

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Auth header for Sage
USER = os.getenv("SAGE_USER", "")
PASSWORD = os.getenv("SAGE_PASSWORD", "")
auth = (USER, PASSWORD)

# Number of concurrent downloads
MAX_WORKERS = 10
DEFAULT_SAMPLE_SIZE = 600

def download_image(session, image_url, filepath, auth):
    """Download and save a single image."""
    try:
        # Skip if file already exists
        if os.path.exists(filepath):
            return {"status": "skipped", "filepath": filepath}
        
        # Ensure directory exists (if filepath has a directory component)
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        # Download image
        response = session.get(image_url, auth=auth, timeout=30)
        response.raise_for_status()
        
        # Save image
        image = Image.open(BytesIO(response.content))
        image.save(filepath)
        
        return {"status": "success", "filepath": filepath}
    except Exception as e:
        logger.error(f"Failed to download {image_url}: {e}")
        return {"status": "error", "filepath": filepath, "error": str(e)}

def extract_path_from_url(url, base_url="https://storage.sagecontinuum.org/api/v1/data"):
    """Extract the path from URL excluding the base URL."""
    if url.startswith(base_url):
        # Remove the base URL and get the path
        path = url[len(base_url):].lstrip('/')
        return path
    else:
        # If URL doesn't match expected format, parse it
        parsed = urlparse(url)
        path = parsed.path.lstrip('/')
        # Remove /api/v1/data prefix if present
        if path.startswith('api/v1/data/'):
            path = path[len('api/v1/data/'):]
        return path

def download_images(df, output_dir, auth, max_workers=MAX_WORKERS):
    """Download images from a dataframe in parallel."""
    # Create session for connection pooling
    session = requests.Session()
    
    # Prepare download tasks
    tasks = []
    for _, row in df.iterrows():
        image_url = row.value
        # Extract path from URL (excluding base URL)
        relative_path = extract_path_from_url(image_url)
        # Combine with output directory
        filepath = os.path.join(output_dir, relative_path)
        # Create directory structure if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tasks.append((image_url, filepath))
    
    logger.info(f"Starting download of {len(tasks)} images to {output_dir}")
    
    # Download in parallel
    results = {"success": 0, "skipped": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(download_image, session, url, path, auth)
            for url, path in tasks
        ]
        
        # Process completed downloads
        for future in as_completed(futures):
            result = future.result()
            results[result["status"]] += 1
    
    session.close()
    logger.info(f"Completed {output_dir}: {results['success']} success, "
                f"{results['skipped']} skipped, {results['error']} errors")
    return results

# Get data points
# 2025-12-10 was an eruption day
logger.info("Querying Sage data...")
fire_mobotix_df = sage_data_client.query(
    start="2025-12-10T06:00:00.000Z",
    end="2025-12-11T05:59:00.000Z", 
    filter={
        "vsn": "W097",
        "task": "imagesampler-mobotix"
    }
)
fire_bottom_df = sage_data_client.query(
    start="2025-12-10T06:00:00.000Z",
    end="2025-12-11T05:59:00.000Z", 
    filter={
        "vsn": "W097",
        "task": "imagesampler-bottom"
    }
)
maybe_fire_mobotix_df = sage_data_client.query(
    start="2025-12-05T06:00:00.000Z",
    end="2025-12-06T05:59:00.000Z", 
    filter={
        "vsn": "W097",
        "task": "imagesampler-mobotix"
    }
)
maybe_fire_bottom_df = sage_data_client.query(
    start="2025-12-05T06:00:00.000Z",
    end="2025-12-06T05:59:00.000Z", 
    filter={
        "vsn": "W097",
        "task": "imagesampler-bottom"
    }
)
logger.info(f"Found {len(fire_mobotix_df)} fire mobotix images and {len(fire_bottom_df)} fire bottom images")
logger.info(f"Found {len(maybe_fire_mobotix_df)} maybe fire mobotix images and {len(maybe_fire_bottom_df)} maybe fire bottom images")

# random sample from mobotix, but keep all "bottom" images since not a lot of them
fire_mobotix_df = fire_mobotix_df.sample(n=DEFAULT_SAMPLE_SIZE, random_state=RANDOM_SEED)
maybe_fire_mobotix_df = maybe_fire_mobotix_df.sample(n=DEFAULT_SAMPLE_SIZE, random_state=RANDOM_SEED)
df = pd.concat([fire_mobotix_df, fire_bottom_df, maybe_fire_mobotix_df, maybe_fire_bottom_df])

# Download images in parallel
logger.info(f"Downloading {len(df)} images...")
download_images(df, "W097", auth)

logger.info("All downloads completed!")
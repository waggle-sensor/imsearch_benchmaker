#!/usr/bin/env python3
"""
get_figlib.py

Download a sample of images from HPWREN FIgLib that will be used for the FireBench benchmark.
"""
import requests
from PIL import Image
from io import BytesIO
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import random
from config import RANDOM_SEED

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INDEX_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html"
MAX_WORKERS = 10
SAMPLE_SIZE = 1300

def download_image(session, image_url, filepath):
    """Download and save a single image."""
    try:
        if os.path.exists(filepath):
            return {"status": "skipped", "filepath": filepath}
        
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        response = session.get(image_url, timeout=30)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image.save(filepath)
        return {"status": "success", "filepath": filepath}
    except Exception as e:
        logger.error(f"Failed to download {image_url}: {e}")
        return {"status": "error", "filepath": filepath, "error": str(e)}

def get_links_from_html(url, session):
    """Extract all links from an HTML page."""
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = []
        skip_hrefs = {'../', './', '/', 'index.html', 'index.htm', '#'}
        
        for link in soup.find_all('a'):
            href = link.get('href')
            if not href or href in skip_hrefs or href.startswith('?'):
                continue
            
            # Skip external links not in HPWREN-FIgLib-Data
            if href.startswith(('mailto:', 'javascript:')):
                continue
            if href.startswith(('http://', 'https://')) and 'HPWREN-FIgLib-Data' not in href:
                continue
            
            full_url = urljoin(url, href)
            if 'HPWREN-FIgLib-Data' in full_url:
                links.append(full_url)
        
        return links
    except Exception as e:
        logger.error(f"Failed to get links from {url}: {e}")
        return []

def is_sequence_directory(url):
    """Check if URL points to a sequence directory (format: YYYYMMDD_firename_camera)."""
    path = urlparse(url).path.rstrip('/')
    if not path:
        return False
    
    parts = [p for p in path.split('/') if p]
    if not parts:
        return False
    
    dirname = parts[-1]
    
    # Skip non-sequence directories
    if dirname in {'Tar', 'Miscellaneous', 'index.html', 'HPWREN-FIgLib-Data'}:
        return False
    
    # Check pattern: starts with 8-digit date, has underscores
    if '_' in dirname:
        first_part = dirname.split('_')[0]
        if len(first_part) == 8 and first_part.isdigit() and len(dirname.split('_')) >= 2:
            return True
    
    return False

def extract_sequence_dir(link):
    """Extract sequence directory URL from a link (handles /index.html)."""
    if link.endswith('/index.html'):
        return link[:-len('/index.html')] + '/'
    return link if link.endswith('/') else link + '/'

def discover_sequences(index_url, session):
    """Discover all fire sequence directories from index.html."""
    logger.info("Discovering fire sequences...")
    links = get_links_from_html(index_url, session)
    
    sequences = []
    for link in links:
        if link.lower().endswith(('.jpg', '.jpeg')):
            continue
        
        dir_url = extract_sequence_dir(link)
        if is_sequence_directory(dir_url):
            sequences.append(dir_url)
    
    logger.info(f"Found {len(sequences)} fire sequences")
    return sequences

def get_images_from_sequence(sequence_url, session):
    """Get all JPEG images from a sequence directory."""
    index_url = sequence_url.rstrip('/') + '/index.html'
    links = get_links_from_html(index_url, session)
    return [link for link in links if link.lower().endswith(('.jpg', '.jpeg'))]

def sample_images(sequences, session, sample_size=SAMPLE_SIZE, random_state=RANDOM_SEED):
    """Collect and sample images from sequences."""
    random.seed(random_state)
    
    logger.info(f"Collecting images from {len(sequences)} sequences...")
    all_images = []
    sequence_map = {}
    
    for i, seq_url in enumerate(sequences, 1):
        images = get_images_from_sequence(seq_url, session)
        for img_url in images:
            all_images.append(img_url)
            sequence_map[img_url] = seq_url
        
        if i % 50 == 0 or i == len(sequences):
            logger.info(f"Processed {i}/{len(sequences)} sequences, found {len(all_images)} images")
    
    logger.info(f"Found {len(all_images)} total images")
    
    if len(all_images) <= sample_size:
        logger.info(f"Using all {len(all_images)} images (less than sample size {sample_size})")
        return all_images, sequence_map
    
    sampled = random.sample(all_images, sample_size)
    logger.info(f"Sampled {sample_size} images from {len(all_images)} total")
    return sampled, sequence_map

def get_filepath(image_url, sequence_map, output_dir):
    """Determine filepath for an image, preserving directory structure."""
    parsed = urlparse(image_url)
    path = parsed.path
    
    if path.startswith('/HPWREN-FIgLib-Data/'):
        relative_path = path[len('/HPWREN-FIgLib-Data/'):]
    else:
        # Fallback: use sequence name + filename
        seq_url = sequence_map.get(image_url, '')
        if seq_url:
            seq_name = urlparse(seq_url).path.rstrip('/').split('/')[-1]
            filename = os.path.basename(path)
            relative_path = os.path.join(seq_name, filename)
        else:
            relative_path = os.path.basename(path)
    
    return os.path.join(output_dir, relative_path)

def download_images(image_urls, sequence_map, output_dir, max_workers=MAX_WORKERS):
    """Download images in parallel."""
    session = requests.Session()
    
    tasks = [
        (url, get_filepath(url, sequence_map, output_dir))
        for url in image_urls
    ]
    
    logger.info(f"Starting download of {len(tasks)} images to {output_dir}")
    results = {"success": 0, "skipped": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_image, session, url, path)
            for url, path in tasks
        ]
        
        for future in as_completed(futures):
            result = future.result()
            results[result["status"]] += 1
    
    session.close()
    logger.info(f"Completed: {results['success']} success, "
                f"{results['skipped']} skipped, {results['error']} errors")
    return results

def main():
    """Main function to download FIgLib images."""
    session = requests.Session()
    
    try:
        sequences = discover_sequences(INDEX_URL, session)
        if not sequences:
            logger.error("No sequences found")
            return
        
        image_urls, sequence_map = sample_images(sequences, session, SAMPLE_SIZE)
        if not image_urls:
            logger.error("No images found")
            return
        
        download_images(image_urls, sequence_map, "HPWREN-FIgLib")
        logger.info("All downloads completed!")
    finally:
        session.close()

if __name__ == "__main__":
    main()

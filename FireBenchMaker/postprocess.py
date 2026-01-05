#!/usr/bin/env python3
"""
postprocess.py

Post-processing commands for dataset analysis and preparation.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import requests
from io import BytesIO
import logging
import sys
from pathlib import Path

# Add parent directory to path to import clipscore package
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from config import RANDOM_SEED, CLIP_ADAPTER, CLIP_ADAPTER_ARGS, HF_REPO_ID, HF_PRIVATE, HF_TOKEN
from clipscore import CLIPAdapterRegistry

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

# -----------------------------
# Graph Generation Functions
# -----------------------------

def generate_image_proportion_donuts(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate donut charts showing image proportions for each categorical column.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the visualization.
    """
    categorical_cols = [
        'viewpoint', 'plume_stage', 'lighting', 'confounder_type',
        'environment_type', 'flame_visible'
    ]
    
    available_categorical = [col for col in categorical_cols if col in df.columns]
    
    if not available_categorical:
        return
    
    logger.info("\nGenerating image proportion donut charts for categorical columns...")
    n_cols = min(3, len(available_categorical))
    n_rows = (len(available_categorical) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if len(available_categorical) > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Color palette for donut charts
    colors = plt.cm.Set3(range(12))
    
    for idx, col in enumerate(available_categorical):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx] if n_rows > 1 or n_cols > 1 else axes[idx]
        
        # Count unique images per category (not qrel rows)
        if 'image_id' in df.columns:
            image_counts = df.groupby(col)['image_id'].nunique().sort_values(ascending=False)
        else:
            # Fallback to row counts if image_id not available
            image_counts = df[col].value_counts()
        
        # Prepare labels - only show label if slice is large enough (> 4% of total)
        total_images = image_counts.sum()
        labels = []
        show_labels = []
        for cat, count in image_counts.items():
            pct = (count / total_images) * 100
            if pct >= 4.0:
                # Shorten category name if too long
                cat_display = str(cat)[:15] + "..." if len(str(cat)) > 15 else str(cat)
                labels.append(f"{cat_display}\n({count}, {pct:.1f}%)")
                show_labels.append(True)
            else:
                labels.append("")
                show_labels.append(False)
        
        # Create donut chart with better label handling
        wedges, texts, autotexts = ax.pie(
            image_counts.values,
            labels=None,  # We'll add labels manually to avoid overlap
            autopct='',  # We'll add percentages manually
            startangle=90,
            colors=colors[:len(image_counts)],
            textprops={'fontsize': 8}
        )
        
        # Draw circle in center to make it a donut
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        
        # Add total count in center
        ax.text(0, 0, f'Total:\n{total_images}\nimages', 
               ha='center', va='center', fontsize=11, fontweight='bold')
        
        ax.set_title(f'Image Proportion: {col}', fontsize=13, fontweight='bold', pad=15)
        
        # Add labels manually with better positioning to avoid overlap
        for i, (wedge, (cat, count)) in enumerate(zip(wedges, image_counts.items())):
            pct = (count / total_images) * 100
            
            if show_labels[i]:  # Only add label for slices >= 4%
                # Calculate angle for this wedge (in degrees)
                angle = (wedge.theta2 + wedge.theta1) / 2
                angle_rad = np.deg2rad(angle)
                
                # Position label outside the donut
                label_radius = 1.25
                x = label_radius * np.cos(angle_rad)
                y = label_radius * np.sin(angle_rad)
                
                # Shorten category name if too long
                cat_display = str(cat)[:20] + "..." if len(str(cat)) > 20 else str(cat)
                
                # Add label with better positioning
                ax.text(x, y, f"{cat_display}\n{count} ({pct:.1f}%)",
                       ha='center', va='center',
                       fontsize=9, fontweight='normal',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray', linewidth=0.5))
                
                # Draw line from donut edge to label
                donut_edge_x = 1.0 * np.cos(angle_rad)
                donut_edge_y = 1.0 * np.sin(angle_rad)
                ax.annotate('', xy=(x, y), xytext=(donut_edge_x, donut_edge_y),
                           arrowprops=dict(arrowstyle='-', color='gray', lw=1.0, alpha=0.6,
                                         connectionstyle='arc3,rad=0'))
            elif pct >= 1.5:  # Show just percentage for medium slices
                # Calculate angle for this wedge
                angle = (wedge.theta2 + wedge.theta1) / 2
                angle_rad = np.deg2rad(angle)
                
                # Position percentage inside the donut
                pct_radius = 0.9
                x = pct_radius * np.cos(angle_rad)
                y = pct_radius * np.sin(angle_rad)
                
                ax.text(x, y, f'{pct:.1f}%',
                       ha='center', va='center',
                       fontsize=8, fontweight='normal')
    
    # Hide empty subplots
    for idx in range(len(available_categorical), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        if n_rows > 1 or n_cols > 1:
            axes[row, col_idx].axis('off')
        else:
            if idx < len(axes):
                axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'image_proportion_donuts.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved image_proportion_donuts.png")


def generate_query_relevancy_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate relevancy distribution per query visualizations.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the visualization.
    """
    if 'query_id' not in df.columns or 'relevance_label' not in df.columns:
        return
    
    logger.info("\nGenerating relevancy distribution per query...")
    query_relevance = df.groupby('query_id')['relevance_label'].agg(['sum', 'count']).reset_index()
    query_relevance['relevance_rate'] = query_relevance['sum'] / query_relevance['count']
    query_relevance = query_relevance.sort_values('relevance_rate', ascending=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Relevance rate per query
    axes[0].bar(range(len(query_relevance)), query_relevance['relevance_rate'], color='coral')
    axes[0].set_title('Relevance Rate per Query', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Query (sorted by relevance rate)', fontsize=12)
    axes[0].set_ylabel('Relevance Rate (relevant / total)', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Distribution of relevance rates
    axes[1].hist(query_relevance['relevance_rate'], bins=20, color='steelblue', edgecolor='black')
    axes[1].set_title('Distribution of Query Relevance Rates', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Relevance Rate', fontsize=12)
    axes[1].set_ylabel('Number of Queries', fontsize=12)
    axes[1].axvline(query_relevance['relevance_rate'].mean(), color='red', linestyle='--',
                   label=f"Mean: {query_relevance['relevance_rate'].mean():.3f}")
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'query_relevancy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved query_relevancy_distribution.png")
    
    # Save table
    query_relevance.to_csv(output_dir / 'query_relevancy_stats.csv', index=False)
    logger.info(f"  ✓ Saved query_relevancy_stats.csv")


def generate_wordclouds(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate word clouds for summaries and tags.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the visualizations.
    """
    logger.info("\nGenerating word clouds...")
    
    # Word cloud for summaries
    if 'summary' in df.columns:
        summaries = df['summary'].dropna().astype(str).tolist()
        if summaries:
            text = ' '.join(summaries)
            # Clean text
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            wordcloud = WordCloud(width=1200, height=600, background_color='white',
                                max_words=100, colormap='viridis').generate(text)
            
            plt.figure(figsize=(15, 8))
            # Convert wordcloud to PIL Image then to numpy array to avoid numpy compatibility issues
            wordcloud_image = wordcloud.to_image()
            plt.imshow(np.array(wordcloud_image), interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud: Image Summaries', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(output_dir / 'wordcloud_summaries.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  ✓ Saved wordcloud_summaries.png")
    
    # Word cloud for tags
    if 'tags' in df.columns:
        all_tags = []
        for tags in df['tags'].dropna():
            if isinstance(tags, list):
                all_tags.extend([str(tag).lower() for tag in tags])
            elif isinstance(tags, str):
                # Try to parse as JSON if it's a string representation
                try:
                    parsed = json.loads(tags)
                    if isinstance(parsed, list):
                        all_tags.extend([str(tag).lower() for tag in parsed])
                except:
                    all_tags.append(tags.lower())
        
        if all_tags:
            tag_freq = Counter(all_tags)
            wordcloud = WordCloud(width=1200, height=600, background_color='white',
                                max_words=100, colormap='plasma').generate_from_frequencies(tag_freq)
            
            plt.figure(figsize=(15, 8))
            # Convert wordcloud to PIL Image then to numpy array to avoid numpy compatibility issues
            wordcloud_image = wordcloud.to_image()
            plt.imshow(np.array(wordcloud_image), interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud: Tags', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(output_dir / 'wordcloud_tags.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  ✓ Saved wordcloud_tags.png")
            
            # Save top tags table
            top_tags = pd.DataFrame(tag_freq.most_common(50), columns=['Tag', 'Count'])
            top_tags.to_csv(output_dir / 'top_tags.csv', index=False)
            logger.info(f"  ✓ Saved top_tags.csv")


def generate_relevance_overview(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate overall relevance distribution visualization.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the visualization.
    """
    if 'relevance_label' not in df.columns:
        return
    
    logger.info("\nGenerating additional visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    rel_counts = df['relevance_label'].value_counts()
    axes[0].pie(rel_counts.values, labels=[f'Not Relevant ({rel_counts.get(0, 0)})', 
                                           f'Relevant ({rel_counts.get(1, 0)})'],
               autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    axes[0].set_title('Overall Relevance Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    rel_counts.plot(kind='bar', ax=axes[1], color=['#ff9999', '#66b3ff'])
    axes[1].set_title('Relevance Label Counts', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Relevance Label', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_xticklabels(['Not Relevant (0)', 'Relevant (1)'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'relevance_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved relevance_overview.png")


def generate_relevance_by_categorical(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate cross-tabulation visualizations: relevance vs categorical features.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the visualizations.
    """
    if 'relevance_label' not in df.columns:
        return
    
    for col in ['viewpoint', 'plume_stage', 'lighting', 'confounder_type', 'environment_type']:
        if col in df.columns:
            crosstab = pd.crosstab(df[col], df['relevance_label'], normalize='index') * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            crosstab.plot(kind='bar', ax=ax, color=['#ff9999', '#66b3ff'], stacked=True)
            ax.set_title(f'Relevance Distribution by {col}', fontsize=14, fontweight='bold')
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Percentage', fontsize=12)
            ax.legend(['Not Relevant (0)', 'Relevant (1)'], title='Relevance Label')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f'relevance_by_{col}.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  ✓ Saved relevance_by_{col}.png")


def generate_query_text_length_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate query text length distribution visualization.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the visualization.
    """
    if 'query_text' not in df.columns:
        return
    
    df['query_text_length'] = df['query_text'].str.len()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['query_text_length'].dropna(), bins=30, color='steelblue', edgecolor='black')
    ax.set_title('Distribution of Query Text Length', fontsize=14, fontweight='bold')
    ax.set_xlabel('Query Text Length (characters)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.axvline(df['query_text_length'].mean(), color='red', linestyle='--',
               label=f"Mean: {df['query_text_length'].mean():.1f} chars")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'query_text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved query_text_length_distribution.png")


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate summary statistics table.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the statistics.
    """
    logger.info("\nGenerating summary statistics...")
    stats = {
        'Total Qrel Rows': len(df),
        'Unique Queries': df['query_id'].nunique() if 'query_id' in df.columns else 'N/A',
        'Unique Images': df['image_id'].nunique() if 'image_id' in df.columns else 'N/A',
        'Relevant Pairs': int(df['relevance_label'].sum()) if 'relevance_label' in df.columns else 'N/A',
        'Not Relevant Pairs': int((df['relevance_label'] == 0).sum()) if 'relevance_label' in df.columns else 'N/A',
        'Relevance Rate': f"{(df['relevance_label'].mean() * 100):.2f}%" if 'relevance_label' in df.columns else 'N/A',
    }
    
    # Add column-specific stats
    for col in ['viewpoint', 'plume_stage', 'lighting', 'confounder_type', 'environment_type']:
        if col in df.columns:
            stats[f'Unique {col} values'] = df[col].nunique()
    
    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    stats_df.to_csv(output_dir / 'dataset_summary_stats.csv', index=False)
    logger.info(f"  ✓ Saved dataset_summary_stats.csv")


def generate_random_image_sample(df: pd.DataFrame, output_dir: Path, images_jsonl_path: Optional[Path] = None) -> None:
    """
    Generate random sample of 50 images visualization.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the visualization.
        images_jsonl_path: Optional path to images.jsonl file for getting image URLs.
    """
    if 'image_id' not in df.columns:
        return
    
    logger.info("\nGenerating random sample of 50 images...")
    
    # Get image URLs from images.jsonl if provided
    image_url_map = {}
    if images_jsonl_path and images_jsonl_path.exists():
        logger.info(f"Loading image URLs from {images_jsonl_path}...")
        images_data = read_jsonl(images_jsonl_path)
        for img_row in images_data:
            img_id = img_row.get('image_id')
            img_url = img_row.get('image_url')
            if img_id and img_url:
                image_url_map[img_id] = img_url
        logger.info(f"Loaded {len(image_url_map)} image URLs")
    elif 'image_url' in df.columns:
        # Fallback to image_url from qrels if available
        for _, row in df[['image_id', 'image_url']].drop_duplicates(subset=['image_id']).iterrows():
            image_url_map[row['image_id']] = row['image_url']
    
    if not image_url_map:
        logger.warning("  ⚠ No image URLs available. Skipping image sample visualization.")
        return
    
    unique_images = df['image_id'].unique()
    # Filter to only images we have URLs for
    available_images = [img_id for img_id in unique_images if img_id in image_url_map]
    sample_size = min(50, len(available_images))
    np.random.seed(RANDOM_SEED)  # For reproducibility
    sampled_image_ids = np.random.choice(available_images, size=sample_size, replace=False)
    
    # Create dataframe with image_id and image_url
    sampled_data = [{'image_id': img_id, 'image_url': image_url_map[img_id]} 
                  for img_id in sampled_image_ids]
    sampled_df = pd.DataFrame(sampled_data)
    
    # Create a grid visualization with actual images
    n_cols = 10
    n_rows = (sample_size + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if sample_size > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, row in sampled_df.iterrows():
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx] if n_rows > 1 or n_cols > 1 else axes[idx]
        
        image_url = row['image_url']
        
        try:
            # Load image from URL
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Display image
            ax.imshow(img)
            ax.axis('off')
        except Exception as e:
            # If image fails to load, show error message
            ax.text(0.5, 0.5, f"Failed to load\n{str(e)[:30]}", 
                   ha='center', va='center', fontsize=8,
                   transform=ax.transAxes, color='red')
            ax.axis('off')
    
    # Hide empty subplots
    for idx in range(sample_size, n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        if n_rows > 1 or n_cols > 1:
            axes[row, col_idx].axis('off')
        else:
            if idx < len(axes):
                axes[idx].axis('off')
    
    plt.suptitle('Random Sample of 50 Images from Dataset', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'random_image_sample.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved random_image_sample.png")


def generate_clipscore_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate CLIP score analysis visualizations.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the visualizations.
    """
    logger.info("\nGenerating CLIP score analysis...")
    
    if 'clip_score' not in df.columns:
        logger.warning("  ⚠ No CLIP scores found. Skipping CLIP score analysis.")
        return
    
    # Filter out None/null clipscore values
    clipscore_df = df[df['clip_score'].notna()].copy()
    
    if len(clipscore_df) == 0:
        logger.warning("  ⚠ No valid CLIP scores found. Skipping CLIP score analysis.")
        return
    
    # 1. CLIP score distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Histogram of CLIP scores
    axes[0, 0].hist(clipscore_df['clip_score'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('CLIP Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('CLIP Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].axvline(clipscore_df['clip_score'].mean(), color='red', linestyle='--',
                       label=f"Mean: {clipscore_df['clip_score'].mean():.3f}")
    axes[0, 0].axvline(clipscore_df['clip_score'].median(), color='green', linestyle='--',
                       label=f"Median: {clipscore_df['clip_score'].median():.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Box plot of CLIP scores by relevance
    if 'relevance_label' in clipscore_df.columns:
        relevant_scores = clipscore_df[clipscore_df['relevance_label'] == 1]['clip_score'].dropna()
        not_relevant_scores = clipscore_df[clipscore_df['relevance_label'] == 0]['clip_score'].dropna()
        
        box_data = [not_relevant_scores, relevant_scores]
        box_labels = ['Not Relevant (0)', 'Relevant (1)']
        
        bp = axes[0, 1].boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('#ff9999')
        bp['boxes'][1].set_facecolor('#66b3ff')
        axes[0, 1].set_title('CLIP Score Distribution by Relevance', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('CLIP Score', fontsize=12)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add mean markers
        if len(not_relevant_scores) > 0:
            axes[0, 1].plot(1, not_relevant_scores.mean(), 'rD', markersize=10, label='Mean')
        if len(relevant_scores) > 0:
            axes[0, 1].plot(2, relevant_scores.mean(), 'rD', markersize=10)
        axes[0, 1].legend(['Mean'], loc='upper right')
    else:
        axes[0, 1].text(0.5, 0.5, 'Relevance label not available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('CLIP Score Distribution by Relevance', fontsize=14, fontweight='bold')
    
    # Plot 3: Violin plot of CLIP scores by relevance (if available)
    if 'relevance_label' in clipscore_df.columns:
        clipscore_df['relevance_label_str'] = clipscore_df['relevance_label'].map({0: 'Not Relevant', 1: 'Relevant'})
        sns.violinplot(data=clipscore_df, x='relevance_label_str', y='clip_score', ax=axes[1, 0],
                      hue='relevance_label_str', palette=['#ff9999', '#66b3ff'], legend=False)
        axes[1, 0].set_title('CLIP Score Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Relevance Label', fontsize=12)
        axes[1, 0].set_ylabel('CLIP Score', fontsize=12)
        axes[1, 0].grid(axis='y', alpha=0.3)
    else:
        axes[1, 0].axis('off')
    
    # Plot 4: Cumulative distribution of CLIP scores
    sorted_scores = np.sort(clipscore_df['clip_score'].dropna())
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1, 1].plot(sorted_scores, cumulative, linewidth=2, color='steelblue')
    axes[1, 1].set_title('Cumulative Distribution of CLIP Scores', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('CLIP Score', fontsize=12)
    axes[1, 1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axvline(clipscore_df['clip_score'].median(), color='red', linestyle='--',
                       label=f"Median: {clipscore_df['clip_score'].median():.3f}")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clipscore_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved clipscore_analysis.png")
    
    # 2. CLIP score statistics by relevance
    if 'relevance_label' in clipscore_df.columns:
        stats_by_relevance = clipscore_df.groupby('relevance_label')['clip_score'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).reset_index()
        stats_by_relevance['relevance_label'] = stats_by_relevance['relevance_label'].map({0: 'Not Relevant', 1: 'Relevant'})
        stats_by_relevance.columns = ['Relevance Label', 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
        stats_by_relevance.to_csv(output_dir / 'clipscore_stats_by_relevance.csv', index=False)
        logger.info(f"  ✓ Saved clipscore_stats_by_relevance.csv")
    
    # 3. Overall CLIP score statistics
    overall_stats = {
        'Total Rows': len(clipscore_df),
        'Mean CLIP Score': f"{clipscore_df['clip_score'].mean():.4f}",
        'Median CLIP Score': f"{clipscore_df['clip_score'].median():.4f}",
        'Std Dev': f"{clipscore_df['clip_score'].std():.4f}",
        'Min CLIP Score': f"{clipscore_df['clip_score'].min():.4f}",
        'Max CLIP Score': f"{clipscore_df['clip_score'].max():.4f}",
        '25th Percentile': f"{clipscore_df['clip_score'].quantile(0.25):.4f}",
        '75th Percentile': f"{clipscore_df['clip_score'].quantile(0.75):.4f}",
    }
    
    if 'relevance_label' in clipscore_df.columns:
        relevant_df = clipscore_df[clipscore_df['relevance_label'] == 1]
        not_relevant_df = clipscore_df[clipscore_df['relevance_label'] == 0]
        
        if len(relevant_df) > 0:
            overall_stats['Mean CLIP Score (Relevant)'] = f"{relevant_df['clip_score'].mean():.4f}"
        if len(not_relevant_df) > 0:
            overall_stats['Mean CLIP Score (Not Relevant)'] = f"{not_relevant_df['clip_score'].mean():.4f}"
    
    stats_df = pd.DataFrame(list(overall_stats.items()), columns=['Metric', 'Value'])
    stats_df.to_csv(output_dir / 'clipscore_overall_stats.csv', index=False)
    logger.info(f"  ✓ Saved clipscore_overall_stats.csv")
    
    # 4. CLIP score vs query (if multiple queries)
    if 'query_id' in clipscore_df.columns:
        query_clipscore_stats = clipscore_df.groupby('query_id')['clip_score'].agg(['mean', 'std', 'count']).reset_index()
        query_clipscore_stats.columns = ['Query ID', 'Mean CLIP Score', 'Std Dev', 'Count']
        query_clipscore_stats = query_clipscore_stats.sort_values('Mean CLIP Score', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(range(len(query_clipscore_stats)), query_clipscore_stats['Mean CLIP Score'], color='coral')
        ax.set_title('Mean CLIP Score per Query', fontsize=14, fontweight='bold')
        ax.set_xlabel('Query (sorted by mean CLIP score)', fontsize=12)
        ax.set_ylabel('Mean CLIP Score', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'clipscore_by_query.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved clipscore_by_query.png")
        
        query_clipscore_stats.to_csv(output_dir / 'clipscore_by_query_stats.csv', index=False)
        logger.info(f"  ✓ Saved clipscore_by_query_stats.csv")


def generate_confidence_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate confidence score analysis visualizations.
    Args:
        df: The dataframe containing the data.
        output_dir: Directory to save the visualizations.
    """
    logger.info("\nGenerating confidence analysis...")
    
    if 'confidence' not in df.columns:
        logger.warning("  ⚠ No confidence column found. Skipping confidence analysis.")
        return
    
    # Filter out None/null confidence values
    confidence_df = df[df['confidence'].notna()].copy()
    
    if len(confidence_df) == 0:
        logger.warning("  ⚠ No valid confidence data found. Skipping confidence analysis.")
        return
    
    # Parse confidence column (might be string JSON or dict)
    confidence_data = []
    for idx, row in confidence_df.iterrows():
        conf = row['confidence']
        if isinstance(conf, str):
            try:
                conf = json.loads(conf)
            except:
                continue
        if isinstance(conf, dict):
            for category, score in conf.items():
                if isinstance(score, (int, float)):
                    confidence_data.append({
                        'row_idx': idx,
                        'category': category,
                        'confidence_score': float(score)
                    })
    
    if not confidence_data:
        logger.warning("  ⚠ No valid confidence scores found. Skipping confidence analysis.")
        return
    
    conf_df = pd.DataFrame(confidence_data)
    
    # Get unique categories
    categories = sorted(conf_df['category'].unique())
    
    if not categories:
        logger.warning("  ⚠ No confidence categories found. Skipping confidence analysis.")
        return
    
    # 1. Confidence distribution by category
    n_cats = len(categories)
    n_cols = min(2, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break
        
        cat_scores = conf_df[conf_df['category'] == category]['confidence_score']
        ax.hist(cat_scores, bins=30, color='teal', edgecolor='black', alpha=0.7)
        ax.set_title(f'Confidence Distribution: {category}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Confidence Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.axvline(cat_scores.mean(), color='red', linestyle='--',
                   label=f"Mean: {cat_scores.mean():.3f}")
        ax.axvline(cat_scores.median(), color='green', linestyle='--',
                   label=f"Median: {cat_scores.median():.3f}")
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(categories), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved confidence_distribution_by_category.png")
    
    # 2. Box plot comparing confidence across categories
    fig, ax = plt.subplots(figsize=(14, 8))
    box_data = [conf_df[conf_df['category'] == cat]['confidence_score'].values for cat in categories]
    bp = ax.boxplot(box_data, tick_labels=categories, patch_artist=True)
    
    colors = plt.cm.Set3(range(len(categories)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('Confidence Scores by Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_by_category_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved confidence_by_category_boxplot.png")
    
    # 3. Overall confidence statistics
    overall_stats = {
        'Total Rows': len(confidence_df),
        'Total Confidence Scores': len(conf_df),
        'Mean Confidence (Overall)': f"{conf_df['confidence_score'].mean():.4f}",
        'Median Confidence (Overall)': f"{conf_df['confidence_score'].median():.4f}",
        'Std Dev (Overall)': f"{conf_df['confidence_score'].std():.4f}",
        'Min Confidence': f"{conf_df['confidence_score'].min():.4f}",
        'Max Confidence': f"{conf_df['confidence_score'].max():.4f}",
        '25th Percentile': f"{conf_df['confidence_score'].quantile(0.25):.4f}",
        '75th Percentile': f"{conf_df['confidence_score'].quantile(0.75):.4f}",
    }
    
    stats_df = pd.DataFrame(list(overall_stats.items()), columns=['Metric', 'Value'])
    stats_df.to_csv(output_dir / 'confidence_overall_stats.csv', index=False)
    logger.info(f"  ✓ Saved confidence_overall_stats.csv")
    
    # 4. Statistics by category
    category_stats = conf_df.groupby('category')['confidence_score'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    category_stats.columns = ['Category', 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
    category_stats = category_stats.sort_values('Mean', ascending=False)
    category_stats.to_csv(output_dir / 'confidence_stats_by_category.csv', index=False)
    logger.info(f"  ✓ Saved confidence_stats_by_category.csv")
    
    # 5. Confidence vs Relevance (if available)
    if 'relevance_label' in confidence_df.columns:
        # Join confidence data with relevance
        conf_with_relevance = []
        for idx, row in confidence_df.iterrows():
            conf = row['confidence']
            relevance = row['relevance_label']
            
            if isinstance(conf, str):
                try:
                    conf = json.loads(conf)
                except:
                    continue
            if isinstance(conf, dict):
                for category, score in conf.items():
                    if isinstance(score, (int, float)):
                        conf_with_relevance.append({
                            'category': category,
                            'confidence_score': float(score),
                            'relevance_label': int(relevance)
                        })
        
        if conf_with_relevance:
            conf_rel_df = pd.DataFrame(conf_with_relevance)
            conf_rel_df['relevance_label_str'] = conf_rel_df['relevance_label'].map({0: 'Not Relevant', 1: 'Relevant'})
            
            # Create comparison plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            relevant_means = []
            not_relevant_means = []
            cat_labels = []
            
            for cat in categories:
                cat_data = conf_rel_df[conf_rel_df['category'] == cat]
                if len(cat_data) > 0:
                    relevant = cat_data[cat_data['relevance_label'] == 1]['confidence_score']
                    not_relevant = cat_data[cat_data['relevance_label'] == 0]['confidence_score']
                    
                    if len(relevant) > 0 and len(not_relevant) > 0:
                        relevant_means.append(relevant.mean())
                        not_relevant_means.append(not_relevant.mean())
                        cat_labels.append(cat)
            
            if relevant_means and not_relevant_means:
                x = np.arange(len(cat_labels))
                width = 0.35
                
                ax.bar(x - width/2, not_relevant_means, width, label='Not Relevant', color='#ff9999', alpha=0.8)
                ax.bar(x + width/2, relevant_means, width, label='Relevant', color='#66b3ff', alpha=0.8)
                
                ax.set_title('Mean Confidence by Category and Relevance', fontsize=14, fontweight='bold')
                ax.set_xlabel('Category', fontsize=12)
                ax.set_ylabel('Mean Confidence Score', fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(cat_labels, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'confidence_by_category_and_relevance.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"  ✓ Saved confidence_by_category_and_relevance.png")


def generate_config_values_table(output_dir: Path) -> None:
    """
    Generate config values CSV table (excluding sensitive information).
    Args:
        output_dir: Directory to save the table.
    """
    logger.info("\nGenerating config values table...")
    try:
        import config as cfg_module
        
        # Define sensitive keys to exclude
        sensitive_keys = ['OPENAI_API_KEY', 'API_KEY', 'KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'HF_TOKEN']
        
        config_data = []
        
        # Get all config variables (excluding private ones starting with _)
        for attr_name in dir(cfg_module):
            if not attr_name.startswith('_') and not callable(getattr(cfg_module, attr_name, None)) and not attr_name.startswith('os'):
                # Skip if contains sensitive keywords
                if any(sensitive in attr_name.upper() for sensitive in sensitive_keys):
                    continue
                
                attr_value = getattr(cfg_module, attr_name)
                config_data.append({
                    'Config Variable': attr_name,
                    'Value': str(attr_value),
                    'Type': type(attr_value).__name__
                })
        
        config_df = pd.DataFrame(config_data)
        config_df = config_df.sort_values('Config Variable')
        config_df.to_csv(output_dir / 'config_values.csv', index=False)
        logger.info(f"  ✓ Saved config_values.csv")
    
    except ImportError:
        logger.warning("Could not import config module. Skipping config values table.")
    except Exception as e:
        logger.warning(f"Error generating config table: {e}")


def generate_dataset_summary(qrels_path: Path, output_dir: Path, images_jsonl_path: Optional[Path] = None) -> None:
    """
    Generate a comprehensive Exploratory Data Analysis (EDA) of the dataset with graphs and tables.
    Args:
        qrels_path: Path to the qrels.jsonl file.
        output_dir: Directory to save the summary visualizations and tables.
        images_jsonl_path: Optional path to images.jsonl file for getting image URLs.
    """
    logger.info("Generating dataset summary...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading qrels from {qrels_path}...")
    qrels = read_jsonl(qrels_path)
    df = pd.DataFrame(qrels)
    
    logger.info(f"Loaded {len(df)} qrel rows")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Generate all visualizations
    generate_image_proportion_donuts(df, output_dir)
    generate_query_relevancy_distribution(df, output_dir)
    generate_wordclouds(df, output_dir)
    generate_relevance_overview(df, output_dir)
    generate_relevance_by_categorical(df, output_dir)
    generate_query_text_length_distribution(df, output_dir)
    generate_clipscore_analysis(df, output_dir)
    generate_confidence_analysis(df, output_dir)
    generate_summary_statistics(df, output_dir)
    generate_random_image_sample(df, output_dir, images_jsonl_path)
    generate_config_values_table(output_dir)
    
    logger.info(f"\n✅ Dataset summary complete! All outputs saved to: {output_dir}")

# -----------------------------
# CLIPScore Calculation Functions
# -----------------------------

def calculate_clipscore(
    qrels_path: Path,
    output_path: Path,
    adapter_name: str,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
    images_jsonl_path: Optional[Path] = None,
    batch_size: int = 1,
    progress_interval: int = 10
) -> None:
    """
    Calculate CLIPScore for each row in qrels JSONL and add it as a column.
    
    Args:
        qrels_path: Path to the input qrels.jsonl file.
        output_path: Path to save the output qrels.jsonl with CLIPScore column.
        adapter_name: Name of the CLIP adapter to use.
        adapter_kwargs: Optional keyword arguments to pass to the adapter constructor.
        batch_size: Number of requests to process before saving progress (currently 1 for sequential processing).
        progress_interval: Log progress every N rows.
    """
    logger.info(f"Calculating CLIPScore using adapter: {adapter_name}")
    
    # Get the adapter instance
    adapter_kwargs = adapter_kwargs or {}
    try:
        adapter = CLIPAdapterRegistry.get(adapter_name, **adapter_kwargs)
        logger.info(f"Initialized adapter: {adapter.get_name()}")
    except Exception as e:
        logger.error(f"Failed to initialize adapter '{adapter_name}': {e}")
        logger.info(f"Available adapters: {CLIPAdapterRegistry.list_adapters()}")
        raise
    
    # Load qrels
    logger.info(f"Loading qrels from {qrels_path}...")
    qrels = read_jsonl(qrels_path)
    logger.info(f"Loaded {len(qrels)} rows")
    
    # Check required columns in qrels
    if not any('query_text' in row or 'query' in row for row in qrels):
        raise ValueError("Missing required column 'query_text' or 'query' in qrels")
    if not any('image_id' in row for row in qrels):
        raise ValueError("Missing required column 'image_id' in qrels")
    
    # Load image URLs from images.jsonl if provided
    image_url_map = {}
    if images_jsonl_path and images_jsonl_path.exists():
        logger.info(f"Loading image URLs from {images_jsonl_path}...")
        images_data = read_jsonl(images_jsonl_path)
        for img_row in images_data:
            img_id = img_row.get('image_id')
            img_url = img_row.get('image_url')
            if img_id and img_url:
                image_url_map[img_id] = img_url
        logger.info(f"Loaded {len(image_url_map)} image URLs from images.jsonl")
    else:
        raise ValueError(
            f"images.jsonl path is required but not provided or file does not exist: {images_jsonl_path}. "
            f"Please provide --images-jsonl argument."
        )
    
    # Process each row
    total_rows = len(qrels)
    successful = 0
    failed = 0
    
    logger.info(f"Processing {total_rows} rows...")
    
    for idx, row in enumerate(qrels):
        try:
            query_text = row.get('query_text') or row.get('query', '')
            image_id = row.get('image_id', '')
            
            if not query_text:
                logger.warning(f"Row {idx + 1}: Missing query_text or query, skipping")
                row['clip_score'] = None
                failed += 1
                continue
            
            if not image_id:
                logger.warning(f"Row {idx + 1}: Missing image_id, skipping")
                row['clip_score'] = None
                failed += 1
                continue
            
            # Get image URL from mapping
            image_url = image_url_map.get(image_id)
            if not image_url:
                logger.warning(f"Row {idx + 1}: image_id '{image_id}' not found in images.jsonl, skipping")
                row['clip_score'] = None
                failed += 1
                continue
            
            # Calculate CLIPScore
            score = adapter.score(query_text, image_url)
            row['clip_score'] = score
            successful += 1
            
            # Log progress
            if (idx + 1) % progress_interval == 0 or (idx + 1) == total_rows:
                logger.info(f"Progress: {idx + 1}/{total_rows} rows processed ({successful} successful, {failed} failed)")
        
        except Exception as e:
            logger.error(f"Row {idx + 1}: Failed to calculate CLIPScore: {e}")
            row['clip_score'] = None
            failed += 1
    
    # Save results
    logger.info(f"Saving results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        for row in qrels:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ CLIPScore calculation complete!")
    logger.info(f"  Total rows: {total_rows}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Output saved to: {output_path}")

# -----------------------------
# Command Functions
# -----------------------------

def cmd_calculate_clipscore(args: argparse.Namespace) -> None:
    """
    Calculate CLIPScore command.
    Args:
        args: The arguments.
    """
    
    # Parse adapter-specific arguments from --adapter-args JSON string
    adapter_name = args.adapter
    adapter_kwargs = {}
    
    if hasattr(args, 'adapter_args') and args.adapter_args:
        try:
            adapter_kwargs = json.loads(args.adapter_args)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse --adapter-args as JSON: {e}")
            raise
    
    # Validate adapter name
    available_adapters = CLIPAdapterRegistry.list_adapters()
    if adapter_name not in available_adapters:
        raise ValueError(
            f"Adapter '{adapter_name}' not found. Available adapters: {', '.join(available_adapters)}. "
            f"Set via --adapter, CLIP_ADAPTER environment variable, or config.CLIP_ADAPTER"
        )
    
    logger.info(f"Using CLIP adapter: {adapter_name}")
    logger.info(f"Adapter args: {adapter_kwargs}")
    
    images_jsonl_path = Path(args.images_jsonl) if hasattr(args, 'images_jsonl') and args.images_jsonl else None
    
    calculate_clipscore(
        qrels_path=Path(args.qrels_jsonl),
        output_path=Path(args.output_jsonl),
        adapter_name=adapter_name,
        adapter_kwargs=adapter_kwargs,
        images_jsonl_path=images_jsonl_path,
        batch_size=getattr(args, 'batch_size', 1),
        progress_interval=getattr(args, 'progress_interval', 10)
    )


def huggingface(
    qrels_path: Path,
    output_dir: Path,
    images_jsonl_path: Optional[Path] = None,
    image_root_dir: Optional[Path] = None,
    progress_interval: int = 100,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = False
) -> None:
    """
    Prepare and optionally upload the dataset to Hugging Face Hub.
    
    This function:
    1. Loads all columns from qrels.jsonl
    2. Either uses local images from image_root_dir OR image URLs from images.jsonl
    3. Creates a Hugging Face dataset with all columns plus images (URLs or local paths)
    4. Saves the dataset locally
    5. Optionally uploads the dataset to Hugging Face Hub (images will be uploaded automatically)
    
    Args:
        qrels_path: Path to the qrels.jsonl file.
        output_dir: Directory to save the prepared dataset.
        images_jsonl_path: Optional path to images.jsonl file containing image_id to image_url mappings.
                          Required if image_root_dir is not provided.
        image_root_dir: Optional root directory for local images. If provided, image_id in qrels is treated
                        as a path relative to this directory. If not provided, uses URLs from images.jsonl.
        progress_interval: Log progress every N rows.
        repo_id: Optional Hugging Face repository ID (e.g., 'username/dataset-name'). If provided, uploads to Hub.
        token: Optional Hugging Face token. If not provided, uses token from `huggingface-cli login`.
        private: Whether to create a private repository (default: False).
    """
    try:
        from datasets import Dataset, Image as HFImage
    except ImportError:
        logger.error("The 'datasets' library is required. Install it with: pip install datasets")
        raise
    
    logger.info("Preparing dataset for Hugging Face upload...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load qrels
    logger.info(f"Loading qrels from {qrels_path}...")
    qrels = read_jsonl(qrels_path)
    logger.info(f"Loaded {len(qrels)} rows from qrels")
    
    # Determine if using local images or URLs
    use_local_images = image_root_dir is not None
    
    if use_local_images:
        logger.info(f"Using local images from: {image_root_dir}")
        image_root_dir = Path(image_root_dir)
        if not image_root_dir.exists():
            raise ValueError(f"Image root directory does not exist: {image_root_dir}")
    else:
        if not images_jsonl_path or not images_jsonl_path.exists():
            raise ValueError(
                "Either --image-root-dir or --images-jsonl must be provided. "
                "If using --images-jsonl, the file must exist."
            )
        logger.info(f"Loading image URLs from {images_jsonl_path}...")
        images_data = read_jsonl(images_jsonl_path)
        image_url_map = {}
        for img_row in images_data:
            img_id = img_row.get('image_id')
            img_url = img_row.get('image_url')
            if img_id and img_url:
                image_url_map[img_id] = img_url
        logger.info(f"Loaded {len(image_url_map)} image URLs")
        
        # Verify all image_ids in qrels have URLs
        missing_images = []
        for row in qrels:
            img_id = row.get('image_id')
            if img_id and img_id not in image_url_map:
                missing_images.append(img_id)
        
        if missing_images:
            logger.warning(f"Found {len(missing_images)} image_ids in qrels without URLs in images.jsonl")
            logger.warning(f"First 10 missing: {missing_images[:10]}")
    
    # Prepare dataset rows - use URLs or local paths directly
    logger.info("Preparing dataset rows...")
    dataset_rows = []
    missing_count = 0
    
    for idx, row in enumerate(qrels):
        img_id = row.get('image_id')
        if not img_id:
            missing_count += 1
            continue
        
        if use_local_images:
            # Use local file path
            source_path = image_root_dir / img_id
            if not source_path.exists():
                logger.warning(f"Image not found: {source_path}")
                missing_count += 1
                continue
            image_path = str(source_path.resolve())
        else:
            # Use URL directly - Hugging Face will handle downloading when needed
            if img_id not in image_url_map:
                logger.warning(f"Image URL not found for: {img_id}")
                missing_count += 1
                continue
            image_path = image_url_map[img_id]
        
        # Create a new row with all columns from qrels plus the image path/URL
        dataset_row = row.copy()
        dataset_row['image'] = image_path
        
        dataset_rows.append(dataset_row)
        
        if (idx + 1) % progress_interval == 0:
            logger.info(f"Prepared {idx + 1}/{len(qrels)} rows...")
    
    if missing_count > 0:
        logger.warning(f"Skipped {missing_count} rows due to missing images")
    
    logger.info(f"Prepared {len(dataset_rows)} dataset rows")
    
    # Create Hugging Face dataset
    logger.info("Creating Hugging Face dataset...")
    
    # Define features for the dataset
    # We'll use Image feature for the image column
    dataset = Dataset.from_list(dataset_rows)
    
    # Cast image column to Image feature
    dataset = dataset.cast_column("image", HFImage())
    
    # Save dataset
    dataset.save_to_disk(str(output_dir / "dataset"))
    logger.info(f"✓ Saved dataset to {output_dir / 'dataset'}")
    
    # Also save metadata
    metadata = {
        "total_rows": len(dataset_rows),
        "total_qrels": len(qrels),
        "missing_images": missing_count,
        "using_local_images": use_local_images,
        "columns": list(dataset_rows[0].keys()) if dataset_rows else []
    }
    
    with open(output_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved dataset metadata to {output_dir / 'dataset_metadata.json'}")
    
    logger.info(f"\n✅ Dataset preparation complete!")
    logger.info(f"   Dataset saved to: {output_dir / 'dataset'}")
    if use_local_images:
        logger.info(f"   Using local images from: {image_root_dir}")
    else:
        logger.info(f"   Using image URLs (will be downloaded during upload if needed)")
    
    # Upload to Hugging Face Hub if repo_id is provided
    if repo_id:
        logger.info(f"\n📤 Uploading dataset to Hugging Face Hub: {repo_id}...")
        try:
            from huggingface_hub import login, HfApi
            
            # Login if token is provided
            if token:
                logger.info("Logging in with provided token...")
                login(token=token)
            else:
                # Try to use existing login (will use token from cache or environment)
                logger.info("Using existing Hugging Face login (run 'huggingface-cli login' if needed)...")
                # Check if we can access the API
                try:
                    api = HfApi()
                    api.whoami()  # Test authentication
                except Exception as e:
                    logger.warning(f"Could not authenticate: {e}")
                    logger.info("Please run 'huggingface-cli login' or provide --token argument")
                    raise
            
            # Push dataset to Hub
            logger.info(f"Pushing dataset to {repo_id}...")
            dataset.push_to_hub(
                repo_id=repo_id,
                private=private
            )
            
            logger.info(f"\n✅ Dataset successfully uploaded to Hugging Face Hub!")
            logger.info(f"   Repository: https://huggingface.co/datasets/{repo_id}")
            
        except ImportError:
            logger.error("The 'huggingface_hub' library is required for uploading. Install it with: pip install huggingface_hub")
            raise
        except Exception as e:
            logger.error(f"Failed to upload dataset to Hugging Face Hub: {e}")
            logger.info(f"Dataset is still saved locally at: {output_dir / 'dataset'}")
            raise
    else:
        logger.info(f"\n   To upload to Hugging Face Hub:")
        logger.info(f"   1. Install: pip install huggingface_hub")
        logger.info(f"   2. Login: huggingface-cli login")
        logger.info(f"   3. Load and push:")
        logger.info(f"      from datasets import load_from_disk")
        logger.info(f"      dataset = load_from_disk('{output_dir / 'dataset'}')")
        logger.info(f"      dataset.push_to_hub('your-username/your-dataset-name')")


def cmd_huggingface(args: argparse.Namespace) -> None:
    """
    Hugging Face command. Prepares and optionally uploads the dataset to Hugging Face Hub.
    Args:
        args: The arguments.
    """
    images_jsonl_path = Path(args.images_jsonl) if hasattr(args, 'images_jsonl') and args.images_jsonl else None
    image_root_dir = Path(args.image_root_dir) if hasattr(args, 'image_root_dir') and args.image_root_dir else None
    
    huggingface(
        qrels_path=Path(args.qrels_jsonl),
        output_dir=Path(args.output_dir),
        images_jsonl_path=images_jsonl_path,
        image_root_dir=image_root_dir,
        progress_interval=getattr(args, 'progress_interval', 100),
        repo_id=getattr(args, 'repo_id', None),
        token=getattr(args, 'token', None),
        private=getattr(args, 'private', False)
    )


def cmd_generate_summary(args: argparse.Namespace) -> None:
    """
    Generate dataset summary command.
    Args:
        args: The arguments.
    """
    images_jsonl_path = Path(args.images_jsonl) if hasattr(args, 'images_jsonl') and args.images_jsonl else None
    generate_dataset_summary(Path(args.qrels_jsonl), Path(args.output_dir), images_jsonl_path)


# -----------------------------
# Parser Functions
# -----------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.
    Args:
        path: The path to the JSONL file.
    Returns:
        A list of dictionaries, one for each line in the file.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser.
    Returns:
        An argument parser.
    """
    parser = argparse.ArgumentParser(description="FireBench post-processing commands")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Generate summary command
    summary_parser = subparsers.add_parser(
        'generate-summary',
        help='Generate comprehensive Exploratory Data Analysis (EDA) of the dataset with graphs and tables'
    )
    summary_parser.add_argument(
        '--qrels-jsonl',
        required=True,
        help='Path to qrels.jsonl file'
    )
    summary_parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save EDA visualizations and tables'
    )
    summary_parser.add_argument(
        '--images-jsonl',
        required=False,
        help='Path to images.jsonl file for getting image URLs (optional)'
    )
    summary_parser.set_defaults(func=cmd_generate_summary)
    
    # Calculate CLIPScore command
    clipscore_parser = subparsers.add_parser(
        'calculate-clipscore',
        help='Calculate CLIPScore for each row in qrels JSONL and add it as a column'
    )
    clipscore_parser.add_argument(
        '--qrels-jsonl',
        required=True,
        help='Path to input qrels.jsonl file'
    )
    clipscore_parser.add_argument(
        '--output-jsonl',
        required=True,
        help='Path to output qrels.jsonl file with CLIPScore column'
    )
    clipscore_parser.add_argument(
        '--images-jsonl',
        required=True,
        help='Path to images.jsonl file containing image_id to image_url mappings'
    )
    clipscore_parser.add_argument(
        '--adapter',
        required=False,
        default=CLIP_ADAPTER,
        choices=CLIPAdapterRegistry.list_adapters(),
        help=f'CLIP adapter to use. Available: {", ".join(CLIPAdapterRegistry.list_adapters())}. '
             f'Can also be set via CLIP_ADAPTER environment variable or config.py. '
             f'Defaults to "local" if not specified.'
    )
    clipscore_parser.add_argument(
        '--adapter-args',
        required=False,
        default=CLIP_ADAPTER_ARGS,
        help='JSON string of keyword arguments to pass to the adapter constructor (e.g., \'{"api_url": "https://api.example.com", "api_key": "key123"}\'). '
             f'Can also be set via CLIP_ADAPTER_ARGS environment variable or config.py. '
             f'Defaults to \'{{"model": "openai/clip-vit-base-patch32", "device": "auto"}}\' if not specified.'
    )
    clipscore_parser.add_argument(
        '--progress-interval',
        type=int,
        default=10,
        help='Log progress every N rows (default: 10)'
    )
    clipscore_parser.set_defaults(func=cmd_calculate_clipscore)
    
    # Hugging Face dataset command
    hf_parser = subparsers.add_parser(
        'huggingface',
        help='Hugging Face dataset command. Prepares and optionally uploads the dataset to Hugging Face Hub.'
    )
    hf_parser.add_argument(
        '--qrels-jsonl',
        required=True,
        help='Path to qrels.jsonl file'
    )
    hf_parser.add_argument(
        '--images-jsonl',
        required=False,
        help='Path to images.jsonl file containing image_id to image_url mappings. Required if --image-root-dir is not provided.'
    )
    hf_parser.add_argument(
        '--image-root-dir',
        required=False,
        help='Root directory for local images. If provided, image_id in qrels is treated as a path relative to this directory. Required if --images-jsonl is not provided.'
    )
    hf_parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save the prepared dataset'
    )
    hf_parser.add_argument(
        '--progress-interval',
        type=int,
        default=100,
        help='Log progress every N rows (default: 100)'
    )
    hf_parser.add_argument(
        '--repo-id',
        required=False,
        default=HF_REPO_ID,
        help='Hugging Face repository ID (e.g., "username/dataset-name"). If provided, uploads the dataset to Hub.'
    )
    hf_parser.add_argument(
        '--token',
        required=False,
        default=HF_TOKEN,
        help='Hugging Face token for authentication. If not provided, uses token from "huggingface-cli login".'
    )
    hf_parser.add_argument(
        '--private',
        action='store_true',
        default=HF_PRIVATE,
        help='Create a private repository (default: False, creates public repository)'
    )
    hf_parser.set_defaults(func=cmd_huggingface)
    
    return parser

# -----------------------------
# Main Function
# -----------------------------

def main() -> None:
    """
    Main function.
    """
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

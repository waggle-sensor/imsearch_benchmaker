"""
postprocess.py

Post-processing utilities for dataset analysis and similarity score calculation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
from wordcloud import WordCloud
from tqdm import tqdm

from .io import read_jsonl
from .config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG
from .scoring import SimilarityAdapterRegistry
from .cost import (
    aggregate_cost_summaries,
    write_cost_summary_csv,
    CostSummary,
)
from .vision import VisionAdapterRegistry
from .judge import JudgeAdapterRegistry
import logging

logger = logging.getLogger(__name__)


def read_jsonl_list(path: Path) -> List[Dict[str, Any]]:
    """
    Read all rows from a JSONL file into a list.
    
    Args:
        path: Path to the JSONL file to read.
        
    Returns:
        List of dictionaries, one per line in the JSONL file.
    """
    return list(read_jsonl(path))


def generate_image_proportion_donuts(df: pd.DataFrame, output_dir: Path, config: BenchmarkConfig) -> None:
    """
    Generate donut charts showing image proportions for each taxonomy column.
    
    Creates donut charts displaying the distribution of images across different
    categories for each taxonomy column in the dataset. Charts are arranged in
    a grid layout with up to 3 columns.
    
    Args:
        df: DataFrame containing the dataset with taxonomy columns.
        output_dir: Directory to save the output PNG file.
        config: BenchmarkConfig instance with column definitions.
    """
    available_categorical = [col for col in config.columns_taxonomy if col in df.columns]
    if not available_categorical:
        return

    n_cols = min(3, len(available_categorical))
    n_rows = (len(available_categorical) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if len(available_categorical) > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.Set3(range(12))
    for idx, col in enumerate(available_categorical):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx] if n_rows > 1 or n_cols > 1 else axes[idx]

        if config.column_image_id in df.columns:
            image_counts = df.groupby(col)[config.column_image_id].nunique().sort_values(ascending=False)
        else:
            image_counts = df[col].value_counts()

        total_images = image_counts.sum()
        labels = []
        show_labels = []
        for cat, count in image_counts.items():
            pct = (count / total_images) * 100
            if pct >= 4.0:
                cat_display = str(cat)[:15] + "..." if len(str(cat)) > 15 else str(cat)
                labels.append(f"{cat_display}\n({count}, {pct:.1f}%)")
                show_labels.append(True)
            else:
                labels.append("")
                show_labels.append(False)

        wedges, _, _ = ax.pie(
            image_counts.values,
            labels=None,
            autopct="",
            startangle=90,
            colors=colors[:len(image_counts)],
            textprops={"fontsize": 8},
        )

        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        ax.add_artist(centre_circle)
        ax.text(0, 0, f"Total:\n{total_images}\nimages",
                ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_title(f"Image Proportion: {col}", fontsize=13, fontweight="bold", pad=15)

        for i, (wedge, (cat, count)) in enumerate(zip(wedges, image_counts.items())):
            pct = (count / total_images) * 100
            if show_labels[i]:
                angle = (wedge.theta2 + wedge.theta1) / 2
                angle_rad = np.deg2rad(angle)
                label_radius = 1.25
                x = label_radius * np.cos(angle_rad)
                y = label_radius * np.sin(angle_rad)
                cat_display = str(cat)[:20] + "..." if len(str(cat)) > 20 else str(cat)
                ax.text(x, y, f"{cat_display}\n{count} ({pct:.1f}%)",
                        ha="center", va="center", fontsize=9, fontweight="normal",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7,
                                  edgecolor="gray", linewidth=0.5))
                donut_edge_x = 1.0 * np.cos(angle_rad)
                donut_edge_y = 1.0 * np.sin(angle_rad)
                ax.annotate("", xy=(x, y), xytext=(donut_edge_x, donut_edge_y),
                            arrowprops=dict(arrowstyle="-", color="gray", lw=1.0, alpha=0.6,
                                            connectionstyle="arc3,rad=0"))
            elif pct >= 1.5:
                angle = (wedge.theta2 + wedge.theta1) / 2
                angle_rad = np.deg2rad(angle)
                pct_radius = 0.9
                x = pct_radius * np.cos(angle_rad)
                y = pct_radius * np.sin(angle_rad)
                ax.text(x, y, f"{pct:.1f}%",
                        ha="center", va="center", fontsize=8, fontweight="normal")

    for idx in range(len(available_categorical), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        if n_rows > 1 or n_cols > 1:
            axes[row, col_idx].axis("off")
        else:
            if idx < len(axes):
                axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "image_proportion_donuts.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Image proportion donuts saved to {output_dir / 'image_proportion_donuts.png'}")

def generate_query_relevancy_distribution(df: pd.DataFrame, output_dir: Path, config: BenchmarkConfig) -> None:
    """
    Generate visualizations and statistics for query relevancy distribution.
    
    Creates bar charts and histograms showing the relevance rate per query and
    the distribution of relevance rates across all queries. Also saves a CSV
    file with per-query relevance statistics.
    
    Args:
        df: DataFrame containing query and relevance data.
        output_dir: Directory to save output PNG and CSV files.
        config: BenchmarkConfig instance with column definitions.
    """
    if config.column_query_id not in df.columns or config.column_relevance not in df.columns:
        return

    query_relevance = df.groupby(config.column_query_id)[config.column_relevance].agg(["sum", "count"]).reset_index()
    query_relevance["relevance_rate"] = query_relevance["sum"] / query_relevance["count"]
    query_relevance = query_relevance.sort_values("relevance_rate", ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].bar(range(len(query_relevance)), query_relevance["relevance_rate"], color="coral")
    axes[0].set_title("Relevance Rate per Query", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Query (sorted by relevance rate)", fontsize=12)
    axes[0].set_ylabel("Relevance Rate (relevant / total)", fontsize=12)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(query_relevance["relevance_rate"], bins=20, color="steelblue", edgecolor="black")
    axes[1].set_title("Distribution of Query Relevance Rates", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Relevance Rate", fontsize=12)
    axes[1].set_ylabel("Number of Queries", fontsize=12)
    axes[1].axvline(query_relevance["relevance_rate"].mean(), color="red", linestyle="--",
                    label=f"Mean: {query_relevance['relevance_rate'].mean():.3f}")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "query_relevancy_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    query_relevance.to_csv(output_dir / "query_relevancy_stats.csv", index=False)
    logger.info(f"Query relevancy distribution saved to {output_dir / 'query_relevancy_distribution.png'}")
    logger.info(f"Query relevancy stats saved to {output_dir / 'query_relevancy_stats.csv'}")

def generate_wordclouds(df: pd.DataFrame, output_dir: Path, config: BenchmarkConfig) -> None:
    """
    Generate word clouds from image summaries and tags.
    
    Creates word cloud visualizations from:
    - Image summaries (if column_summary is available)
    - Image tags (if column_tags is available)
    
    Also saves a CSV file with the top 50 most frequent tags.
    
    Args:
        df: DataFrame containing summary and tag data.
        output_dir: Directory to save output PNG and CSV files.
        config: BenchmarkConfig instance with column definitions.
    """
    if config.column_summary and config.column_summary in df.columns:
        summaries = df[config.column_summary].dropna().astype(str).tolist()
        if summaries:
            text = " ".join(summaries)
            text = re.sub(r"[^\w\s]", "", text.lower())
            wordcloud = WordCloud(width=1200, height=600, background_color="white",
                                  max_words=100, colormap="viridis").generate(text)
            plt.figure(figsize=(15, 8))
            wordcloud_image = wordcloud.to_image()
            plt.imshow(np.array(wordcloud_image), interpolation="bilinear")
            plt.axis("off")
            plt.title("Word Cloud: Image Summaries", fontsize=16, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(output_dir / "wordcloud_summaries.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Word cloud for summaries saved to {output_dir / 'wordcloud_summaries.png'}")

    if config.column_tags and config.column_tags in df.columns:
        all_tags = []
        for tags in df[config.column_tags].dropna():
            if isinstance(tags, list):
                all_tags.extend([str(tag).lower() for tag in tags])
            elif isinstance(tags, str):
                try:
                    parsed = json.loads(tags)
                    if isinstance(parsed, list):
                        all_tags.extend([str(tag).lower() for tag in parsed])
                except json.JSONDecodeError:
                    all_tags.append(tags.lower())

        if all_tags:
            tag_freq = Counter(all_tags)
            wordcloud = WordCloud(width=1200, height=600, background_color="white",
                                  max_words=100, colormap="plasma").generate_from_frequencies(tag_freq)
            plt.figure(figsize=(15, 8))
            wordcloud_image = wordcloud.to_image()
            plt.imshow(np.array(wordcloud_image), interpolation="bilinear")
            plt.axis("off")
            plt.title("Word Cloud: Tags", fontsize=16, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(output_dir / "wordcloud_tags.png", dpi=300, bbox_inches="tight")
            plt.close()
            top_tags = pd.DataFrame(tag_freq.most_common(50), columns=["Tag", "Count"])
            top_tags.to_csv(output_dir / "top_tags.csv", index=False)
            logger.info(f"Word cloud for tags saved to {output_dir / 'wordcloud_tags.png'}")
            logger.info(f"Top tags saved to {output_dir / 'top_tags.csv'}")

def generate_relevance_overview(df: pd.DataFrame, output_dir: Path, config: BenchmarkConfig) -> None:
    """
    Generate overview visualizations of relevance distribution.
    
    Creates a pie chart and bar chart showing the overall distribution of
    relevant vs. not relevant labels in the dataset.
    
    Args:
        df: DataFrame containing relevance data.
        output_dir: Directory to save the output PNG file.
        config: BenchmarkConfig instance with column definitions.
    """
    if config.column_relevance not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rel_counts = df[config.column_relevance].value_counts()
    axes[0].pie(rel_counts.values, labels=[f"Not Relevant ({rel_counts.get(0, 0)})",
                                           f"Relevant ({rel_counts.get(1, 0)})"],
                autopct="%1.1f%%", startangle=90, colors=["#ff9999", "#66b3ff"])
    axes[0].set_title("Overall Relevance Distribution", fontsize=14, fontweight="bold")

    rel_counts.plot(kind="bar", ax=axes[1], color=["#ff9999", "#66b3ff"])
    axes[1].set_title("Relevance Label Counts", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Relevance Label", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_xticklabels(["Not Relevant (0)", "Relevant (1)"], rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "relevance_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Relevance overview saved to {output_dir / 'relevance_overview.png'}")


def generate_relevance_by_categorical(df: pd.DataFrame, output_dir: Path, config: BenchmarkConfig) -> None:
    """
    Generate stacked bar charts showing relevance distribution by taxonomy categories.
    
    Creates a separate visualization for each taxonomy column, showing the percentage
    of relevant vs. not relevant images within each category.
    
    Args:
        df: DataFrame containing relevance and taxonomy data.
        output_dir: Directory to save output PNG files (one per taxonomy column).
        config: BenchmarkConfig instance with column definitions.
    """
    if config.column_relevance not in df.columns:
        return

    for col in config.columns_taxonomy:
        if col in df.columns:
            crosstab = pd.crosstab(df[col], df[config.column_relevance], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(12, 6))
            crosstab.plot(kind="bar", ax=ax, color=["#ff9999", "#66b3ff"], stacked=True)
            ax.set_title(f"Relevance Distribution by {col}", fontsize=14, fontweight="bold")
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("Percentage", fontsize=12)
            ax.legend(["Not Relevant (0)", "Relevant (1)"], title="Relevance Label")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f"relevance_by_{col}.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Relevance by {col} saved to {output_dir / f'relevance_by_{col}.png'}")

def generate_query_text_length_distribution(df: pd.DataFrame, output_dir: Path, config: BenchmarkConfig) -> None:
    """
    Generate histogram showing the distribution of query text lengths.
    
    Creates a histogram with mean line showing the character length distribution
    of query texts in the dataset.
    
    Args:
        df: DataFrame containing query text data.
        output_dir: Directory to save the output PNG file.
        config: BenchmarkConfig instance with column definitions.
    """
    if config.column_query not in df.columns:
        return

    df["query_text_length"] = df[config.column_query].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df["query_text_length"].dropna(), bins=30, color="steelblue", edgecolor="black")
    ax.set_title("Distribution of Query Text Length", fontsize=14, fontweight="bold")
    ax.set_xlabel("Query Text Length (characters)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.axvline(df["query_text_length"].mean(), color="red", linestyle="--",
               label=f"Mean: {df['query_text_length'].mean():.1f} chars")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "query_text_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Query text length distribution saved to {output_dir / 'query_text_length_distribution.png'}")


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path, config: BenchmarkConfig) -> None:
    """
    Generate summary statistics CSV file for the dataset.
    
    Calculates and saves key statistics including:
    - Total qrel rows, unique queries, unique images
    - Relevant/not relevant pair counts and relevance rate
    - Unique values for each taxonomy column
    
    Args:
        df: DataFrame containing the dataset.
        output_dir: Directory to save the output CSV file.
        config: BenchmarkConfig instance with column definitions.
    """
    stats = {
        "Total Qrel Rows": len(df),
        "Unique Queries": df[config.column_query_id].nunique() if config.column_query_id in df.columns else "N/A",
        "Unique Images": df[config.column_image_id].nunique() if config.column_image_id in df.columns else "N/A",
        "Relevant Pairs": int(df[config.column_relevance].sum()) if config.column_relevance in df.columns else "N/A",
        "Not Relevant Pairs": int((df[config.column_relevance] == 0).sum()) if config.column_relevance in df.columns else "N/A",
        "Relevance Rate": f"{(df[config.column_relevance].mean() * 100):.2f}%" if config.column_relevance in df.columns else "N/A",
    }

    for col in config.columns_taxonomy:
        if col in df.columns:
            stats[f"Unique {col} values"] = df[col].nunique()

    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(output_dir / "dataset_summary_stats.csv", index=False)
    logger.info(f"Dataset summary statistics saved to {output_dir / 'dataset_summary_stats.csv'}")


def generate_random_image_sample(
    df: pd.DataFrame,
    output_dir: Path,
    images_jsonl_path: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
) -> None:
    """
    Generate a grid visualization of randomly sampled images from the dataset.
    
    Samples up to 50 random images from the dataset and displays them in a grid
    layout. Images are loaded from URLs or local paths. Failed image loads are
    indicated with error messages.
    
    Args:
        df: DataFrame containing image IDs.
        output_dir: Directory to save the output PNG file.
        images_jsonl_path: Optional path to images.jsonl for URL mapping.
        config: Optional BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    if config.column_image_id not in df.columns:
        return

    image_url_map: Dict[str, str] = {}
    if images_jsonl_path and images_jsonl_path.exists():
        images_data = read_jsonl_list(images_jsonl_path)
        for img_row in images_data:
            img_id = img_row.get(config.column_image_id)
            img_url = img_row.get(config.image_url_temp_column)
            if img_id and img_url:
                image_url_map[img_id] = img_url
    elif config.image_url_temp_column in df.columns:
        cols = [config.column_image_id, config.image_url_temp_column]
        for _, row in df[cols].drop_duplicates(subset=[config.column_image_id]).iterrows():
            image_url_map[row[config.column_image_id]] = row[config.image_url_temp_column]

    if not image_url_map:
        return

    unique_images = df[config.column_image_id].unique()
    available_images = [img_id for img_id in unique_images if img_id in image_url_map]
    sample_size = min(50, len(available_images))
    np.random.seed(42)
    sampled_image_ids = np.random.choice(available_images, size=sample_size, replace=False)

    sampled_data = [{"image_id": img_id, "image_url": image_url_map[img_id]} for img_id in sampled_image_ids]
    sampled_df = pd.DataFrame(sampled_data)

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
        image_url = row["image_url"]
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            ax.imshow(img)
            ax.axis("off")
        except Exception as exc:
            ax.text(0.5, 0.5, f"Failed to load\n{str(exc)[:30]}",
                    ha="center", va="center", fontsize=8, transform=ax.transAxes, color="red")
            ax.axis("off")

    for idx in range(sample_size, n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        if n_rows > 1 or n_cols > 1:
            axes[row, col_idx].axis("off")
        else:
            if idx < len(axes):
                axes[idx].axis("off")

    plt.suptitle("Random Sample of 50 Images from Dataset", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "random_image_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Random image sample saved to {output_dir / 'random_image_sample.png'}")


def generate_similarity_score_analysis(df: pd.DataFrame, output_dir: Path, config: BenchmarkConfig) -> None:
    """
    Generate comprehensive analysis of similarity scores in the dataset.
    
    Creates multiple visualizations and statistics:
    - Histogram of similarity score distribution
    - Box plots comparing scores by relevance label
    - Violin plots showing score distributions
    - Cumulative distribution function
    - Per-query similarity score statistics
    
    Also saves CSV files with statistics by relevance and overall statistics.
    
    Args:
        df: DataFrame containing similarity scores and relevance data.
        output_dir: Directory to save output PNG and CSV files.
        config: BenchmarkConfig instance with column definitions.
    """
    col_name = config.similarity_config.col_name
    if col_name not in df.columns:
        return

    similarity_score_df = df[df[col_name].notna()].copy()
    if len(similarity_score_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].hist(similarity_score_df[col_name].dropna(), bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title(f"{col_name} Distribution", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel(col_name, fontsize=12)
    axes[0, 0].set_ylabel("Frequency", fontsize=12)
    axes[0, 0].axvline(similarity_score_df[col_name].mean(), color="red", linestyle="--",
                       label=f"Mean: {similarity_score_df[col_name].mean():.3f}")
    axes[0, 0].axvline(similarity_score_df[col_name].median(), color="green", linestyle="--",
                       label=f"Median: {similarity_score_df[col_name].median():.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    if config.column_relevance in similarity_score_df.columns:  
        relevant_scores = similarity_score_df[similarity_score_df[config.column_relevance] == 1][col_name].dropna()
        not_relevant_scores = similarity_score_df[similarity_score_df[config.column_relevance] == 0][col_name].dropna()
        box_data = [not_relevant_scores, relevant_scores]
        box_labels = ["Not Relevant (0)", "Relevant (1)"]
        bp = axes[0, 1].boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        bp["boxes"][0].set_facecolor("#ff9999")
        bp["boxes"][1].set_facecolor("#66b3ff")
        axes[0, 1].set_title(f"{col_name} Distribution by Relevance", fontsize=14, fontweight="bold")
        axes[0, 1].set_ylabel(col_name, fontsize=12)
        axes[0, 1].grid(axis="y", alpha=0.3)
        if len(not_relevant_scores) > 0:
            axes[0, 1].plot(1, not_relevant_scores.mean(), "rD", markersize=10, label="Mean")
        if len(relevant_scores) > 0:
            axes[0, 1].plot(2, relevant_scores.mean(), "rD", markersize=10)
        axes[0, 1].legend(["Mean"], loc="upper right")
    else:
        axes[0, 1].text(0.5, 0.5, "Relevance label not available",
                        ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_title(f"{col_name} Distribution by Relevance", fontsize=14, fontweight="bold")

    if config.column_relevance in similarity_score_df.columns:
        similarity_score_df["relevance_label_str"] = similarity_score_df[config.column_relevance].map({0: "Not Relevant", 1: "Relevant"})
        sns.violinplot(data=similarity_score_df, x="relevance_label_str", y=col_name, ax=axes[1, 0],
                       hue="relevance_label_str", palette=["#ff9999", "#66b3ff"], legend=False)
        axes[1, 0].set_title(f"{col_name} Distribution (Violin Plot)", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Relevance Label", fontsize=12)
        axes[1, 0].set_ylabel(col_name, fontsize=12)
        axes[1, 0].grid(axis="y", alpha=0.3)
    else:
        axes[1, 0].axis("off")

    sorted_scores = np.sort(similarity_score_df[col_name].dropna())
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1, 1].plot(sorted_scores, cumulative, linewidth=2, color="steelblue")
    axes[1, 1].set_title(f"Cumulative Distribution of {col_name}", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel(col_name, fontsize=12)
    axes[1, 1].set_ylabel("Cumulative Probability", fontsize=12)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axvline(similarity_score_df[col_name].median(), color="red", linestyle="--",
                       label=f"Median: {similarity_score_df[col_name].median():.3f}")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"{col_name}_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    if config.column_relevance in similarity_score_df.columns:
        stats_by_relevance = similarity_score_df.groupby(config.column_relevance)[col_name].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).reset_index()
        stats_by_relevance[config.column_relevance] = stats_by_relevance[config.column_relevance].map({0: "Not Relevant", 1: "Relevant"})
        stats_by_relevance.columns = ["Relevance Label", "Count", "Mean", "Median", "Std Dev", "Min", "Max"]
        stats_by_relevance.to_csv(output_dir / f"{col_name}_stats_by_relevance.csv", index=False)
        logger.info(f"Similarity score statistics by relevance saved to {output_dir / f'{col_name}_stats_by_relevance.csv'}")

    overall_stats = {
        "Total Rows": len(similarity_score_df),
        "Mean {col_name}": f"{similarity_score_df[col_name].mean():.4f}",
        "Median {col_name}": f"{similarity_score_df[col_name].median():.4f}",
        "Std Dev": f"{similarity_score_df[col_name].std():.4f}",
        "Min {col_name}": f"{similarity_score_df[col_name].min():.4f}",
        "Max {col_name}": f"{similarity_score_df[col_name].max():.4f}",
        "25th Percentile": f"{similarity_score_df[col_name].quantile(0.25):.4f}",
        "75th Percentile": f"{similarity_score_df[col_name].quantile(0.75):.4f}",
    }

    if config.column_relevance in similarity_score_df.columns:
        relevant_df = similarity_score_df[similarity_score_df[config.column_relevance] == 1]
        not_relevant_df = similarity_score_df[similarity_score_df[config.column_relevance] == 0]
        if len(relevant_df) > 0:
            overall_stats["Mean {col_name} (Relevant)"] = f"{relevant_df[col_name].mean():.4f}"
        if len(not_relevant_df) > 0:
            overall_stats["Mean {col_name} (Not Relevant)"] = f"{not_relevant_df[col_name].mean():.4f}"

    stats_df = pd.DataFrame(list(overall_stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(output_dir / f"{col_name}_overall_stats.csv", index=False)
    logger.info(f"Similarity score overall statistics saved to {output_dir / f'{col_name}_overall_stats.csv'}")

    if config.column_query_id in similarity_score_df.columns:
        query_similarity_score_stats = similarity_score_df.groupby(config.column_query_id)[col_name].agg(["mean", "std", "count"]).reset_index()
        query_similarity_score_stats.columns = ["Query ID", "Mean {col_name}", "Std Dev", "Count"]
        query_similarity_score_stats = query_similarity_score_stats.sort_values("Mean {col_name}", ascending=False)
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(range(len(query_similarity_score_stats)), query_similarity_score_stats["Mean {col_name}"], color="coral")
        ax.set_title(f"Mean {col_name} per Query", fontsize=14, fontweight="bold")
        ax.set_xlabel("Query (sorted by mean {col_name})", fontsize=12)
        ax.set_ylabel(f"Mean {col_name}", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{col_name}_by_query.png", dpi=300, bbox_inches="tight")
        plt.close()
        query_similarity_score_stats.to_csv(output_dir / f"{col_name}_by_query_stats.csv", index=False)
        logger.info(f"Similarity score statistics by query saved to {output_dir / f'{col_name}_by_query_stats.csv'}")

def generate_confidence_analysis(df: pd.DataFrame, output_dir: Path, config: BenchmarkConfig) -> None:
    """
    Generate analysis of confidence scores for taxonomy and boolean columns.
    
    Creates visualizations and statistics for confidence scores:
    - Histograms of confidence distribution per category
    - Box plots comparing confidence across categories
    - Bar charts comparing mean confidence by category and relevance label
    
    Also saves CSV files with overall and per-category confidence statistics.
    
    Args:
        df: DataFrame containing confidence score data.
        output_dir: Directory to save output PNG and CSV files.
        config: BenchmarkConfig instance with column definitions.
    """
    if not config.column_confidence or config.column_confidence not in df.columns:
        return

    confidence_df = df[df[config.column_confidence].notna()].copy()
    if len(confidence_df) == 0:
        return

    confidence_data = []
    for idx, row in confidence_df.iterrows():
        conf = row[config.column_confidence]
        if isinstance(conf, str):
            try:
                conf = json.loads(conf)
            except json.JSONDecodeError:
                continue
        if isinstance(conf, dict):
            for category, score in conf.items():
                if isinstance(score, (int, float)):
                    confidence_data.append({
                        "row_idx": idx,
                        "category": category,
                        "confidence_score": float(score),
                    })

    if not confidence_data:
        return

    conf_df = pd.DataFrame(confidence_data)
    categories = sorted(conf_df["category"].unique())
    if not categories:
        return

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
        cat_scores = conf_df[conf_df["category"] == category]["confidence_score"]
        ax.hist(cat_scores, bins=30, color="teal", edgecolor="black", alpha=0.7)
        ax.set_title(f"Confidence Distribution: {category}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence Score", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.axvline(cat_scores.mean(), color="red", linestyle="--",
                   label=f"Mean: {cat_scores.mean():.3f}")
        ax.axvline(cat_scores.median(), color="green", linestyle="--",
                   label=f"Median: {cat_scores.median():.3f}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(categories), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution_by_category.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 8))
    box_data = [conf_df[conf_df["category"] == cat]["confidence_score"].values for cat in categories]
    bp = ax.boxplot(box_data, tick_labels=categories, patch_artist=True)
    colors = plt.cm.Set3(range(len(categories)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_title("Confidence Scores by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Confidence Score", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_by_category_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    overall_stats = {
        "Total Rows": len(confidence_df),
        "Total Confidence Scores": len(conf_df),
        "Mean Confidence (Overall)": f"{conf_df['confidence_score'].mean():.4f}",
        "Median Confidence (Overall)": f"{conf_df['confidence_score'].median():.4f}",
        "Std Dev (Overall)": f"{conf_df['confidence_score'].std():.4f}",
        "Min Confidence": f"{conf_df['confidence_score'].min():.4f}",
        "Max Confidence": f"{conf_df['confidence_score'].max():.4f}",
        "25th Percentile": f"{conf_df['confidence_score'].quantile(0.25):.4f}",
        "75th Percentile": f"{conf_df['confidence_score'].quantile(0.75):.4f}",
    }
    stats_df = pd.DataFrame(list(overall_stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(output_dir / "confidence_overall_stats.csv", index=False)

    category_stats = conf_df.groupby("category")["confidence_score"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reset_index()
    category_stats.columns = ["Category", "Count", "Mean", "Median", "Std Dev", "Min", "Max"]
    category_stats = category_stats.sort_values("Mean", ascending=False)
    category_stats.to_csv(output_dir / "confidence_stats_by_category.csv", index=False)

    if config.column_relevance in confidence_df.columns:
        conf_with_relevance = []
        for _, row in confidence_df.iterrows():
            conf = row[config.column_confidence]
            relevance = row[config.column_relevance]
            if isinstance(conf, str):
                try:
                    conf = json.loads(conf)
                except json.JSONDecodeError:
                    continue
            if isinstance(conf, dict):
                for category, score in conf.items():
                    if isinstance(score, (int, float)):
                        conf_with_relevance.append({
                            "category": category,
                            "confidence_score": float(score),
                            "relevance_label": int(relevance),
                        })

        if conf_with_relevance:
            conf_rel_df = pd.DataFrame(conf_with_relevance)
            conf_rel_df["relevance_label_str"] = conf_rel_df["relevance_label"].map({0: "Not Relevant", 1: "Relevant"})

            fig, ax = plt.subplots(figsize=(14, 8))
            relevant_means = []
            not_relevant_means = []
            cat_labels = []
            for cat in categories:
                cat_data = conf_rel_df[conf_rel_df["category"] == cat]
                if len(cat_data) > 0:
                    relevant = cat_data[cat_data["relevance_label"] == 1]["confidence_score"]
                    not_relevant = cat_data[cat_data["relevance_label"] == 0]["confidence_score"]
                    if len(relevant) > 0 and len(not_relevant) > 0:
                        relevant_means.append(relevant.mean())
                        not_relevant_means.append(not_relevant.mean())
                        cat_labels.append(cat)

            if relevant_means and not_relevant_means:
                x = np.arange(len(cat_labels))
                width = 0.35
                ax.bar(x - width / 2, not_relevant_means, width, label="Not Relevant", color="#ff9999", alpha=0.8)
                ax.bar(x + width / 2, relevant_means, width, label="Relevant", color="#66b3ff", alpha=0.8)
                ax.set_title("Mean Confidence by Category and Relevance", fontsize=14, fontweight="bold")
                ax.set_xlabel("Category", fontsize=12)
                ax.set_ylabel("Mean Confidence Score", fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(cat_labels, rotation=45, ha="right")
                ax.legend()
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "confidence_by_category_and_relevance.png", dpi=300, bbox_inches="tight")
                plt.close()
    logger.info(f"Confidence analysis saved to {output_dir / 'confidence_by_category_and_relevance.png'}")

def generate_config_values_table(
    output_dir: Path,
    config: BenchmarkConfig,
) -> None:
    """
    Generate a CSV table of config values using config.to_csv().
    Automatically excludes fields starting with underscore (_).
    """
    csv_content = config.to_csv()
    (output_dir / "config_values.csv").write_text(csv_content)
    logger.info(f"Config values table saved to {output_dir / 'config_values.csv'}")


def generate_dataset_summary(
    qrels_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    images_jsonl_path: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
) -> None:
    """
    Generate a comprehensive Exploratory Data Analysis (EDA) of the dataset.
    
    Args:
        qrels_path: Path to the qrels.jsonl file. If None, uses config.qrels_with_score_jsonl or config.qrels_jsonl.
        output_dir: Directory to save the summary. If None, uses config.summary_output_dir.
        images_jsonl_path: Optional path to images.jsonl file. If None, uses config.images_jsonl.
        config: Optional BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    # Get paths from config if not provided
    qrels_path = Path(qrels_path) if qrels_path else (Path(config.qrels_with_score_jsonl) if config.qrels_with_score_jsonl else (Path(config.qrels_jsonl) if config.qrels_jsonl else None))
    output_dir = Path(output_dir) if output_dir else (Path(config.summary_output_dir) if config.summary_output_dir else None)
    images_jsonl_path = Path(images_jsonl_path) if images_jsonl_path else (Path(config.images_jsonl) if config.images_jsonl else None)
    
    if qrels_path is None:
        raise ValueError("qrels_path must be provided or set in config.qrels_jsonl or config.qrels_with_score_jsonl")
    if output_dir is None:
        raise ValueError("output_dir must be provided or set in config.summary_output_dir")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    qrels = read_jsonl_list(qrels_path)
    df = pd.DataFrame(qrels)
    missing_required = [col for col in config.required_qrels_columns() if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in qrels: {missing_required}")

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # List of generation functions with their descriptions
    generation_tasks = [
        ("Image proportion donuts", generate_image_proportion_donuts, (df, output_dir, config)),
        ("Query relevancy distribution", generate_query_relevancy_distribution, (df, output_dir, config)),
        ("Word clouds", generate_wordclouds, (df, output_dir, config)),
        ("Relevance overview", generate_relevance_overview, (df, output_dir, config)),
        ("Relevance by categorical", generate_relevance_by_categorical, (df, output_dir, config)),
        ("Query text length distribution", generate_query_text_length_distribution, (df, output_dir, config)),
        ("Similarity score analysis", generate_similarity_score_analysis, (df, output_dir, config)),
        ("Confidence analysis", generate_confidence_analysis, (df, output_dir, config)),
        ("Summary statistics", generate_summary_statistics, (df, output_dir, config)),
        ("Random image sample", generate_random_image_sample, (df, output_dir, images_jsonl_path, config)),
        ("Config values table", generate_config_values_table, (output_dir, config)),
        ("Cost summary", _generate_cost_summary, (output_dir, config)),
    ]
    
    # Use tqdm to show progress through generation tasks
    with tqdm(total=len(generation_tasks), desc="Generating dataset summary", unit="task") as pbar:
        for task_name, task_func, task_args in generation_tasks:
            try:
                task_func(*task_args)
                pbar.set_postfix({"current": task_name})
            except Exception as e:
                logger.warning(f"Failed to generate {task_name}: {e}")
            finally:
                pbar.update(1)

def _generate_cost_summary(
    output_dir: Path,
    config: BenchmarkConfig,
) -> None:
    """
    Generate cost summary CSV for vision and judge phases.
    
    This is a helper function called by generate_dataset_summary to include
    cost information in the dataset summary output.
    
    Args:
        output_dir: Directory to save cost summary CSV.
        config: BenchmarkConfig instance.
    """
    # Determine batch output file paths
    vision_batch_output_jsonl = None
    if config.annotations_jsonl:
        vision_batch_output_jsonl = Path(config.annotations_jsonl).parent / "vision_batch_output.jsonl"
    
    judge_batch_output_jsonl = None
    if config.qrels_jsonl:
        judge_batch_output_jsonl = Path(config.qrels_jsonl).parent / "judge_batch_output.jsonl"
    
    summaries = []
    
    # Calculate vision costs
    if vision_batch_output_jsonl and vision_batch_output_jsonl.exists():
        try:
            vision_adapter_name = config.vision_config.adapter
            if vision_adapter_name:
                vision_adapter = VisionAdapterRegistry.get(vision_adapter_name, config=config)
                vision_summary = vision_adapter.calculate_actual_costs(
                    batch_output_jsonl=vision_batch_output_jsonl,
                )
                if vision_summary.num_items > 0:
                    summaries.append(vision_summary)
                    logger.info(f"Vision costs: ${vision_summary.total_cost:.2f} for {vision_summary.num_items} images")
        except Exception as e:
            logger.warning(f"Failed to calculate vision costs: {e}")
    
    # Calculate judge costs
    if judge_batch_output_jsonl and judge_batch_output_jsonl.exists():
        try:
            judge_adapter_name = config.judge_config.adapter
            if judge_adapter_name:
                judge_adapter = JudgeAdapterRegistry.get(judge_adapter_name, config=config)
                judge_summary = judge_adapter.calculate_actual_costs(
                    batch_output_jsonl=judge_batch_output_jsonl,
                )
                if judge_summary.num_items > 0:
                    summaries.append(judge_summary)
                    logger.info(f"Judge costs: ${judge_summary.total_cost:.2f} for {judge_summary.num_items} queries")
        except Exception as e:
            logger.warning(f"Failed to calculate judge costs: {e}")
    
    # Aggregate and write CSV
    if summaries:
        total_summary = aggregate_cost_summaries(summaries)
        summaries.append(total_summary)
        
        csv_path = output_dir / "cost_summary.csv"
        write_cost_summary_csv(summaries, csv_path)
        logger.info(f"Cost summary written to {csv_path}")
        logger.info(f"Total cost: ${total_summary.total_cost:.2f}")


def calculate_similarity_score(
    qrels_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    col_name: Optional[str] = None,
    adapter_name: Optional[str] = None,
    images_jsonl_path: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
) -> None:
    """
    Calculate similarity score (e.g., CLIPScore) for each row in qrels JSONL and add it as a column.
    
    Args:
        qrels_path: Path to the input qrels.jsonl file. If None, uses config.qrels_jsonl.
        output_path: Path to save the output qrels.jsonl with similarity score column. If None, uses config.qrels_with_score_jsonl.
        col_name: Name of the column to add the similarity score to. If None, uses config.similarity_config.col_name or defaults to "similarity_score".
        adapter_name: Optional similarity adapter name. If None, uses config.similarity_config.adapter.
        images_jsonl_path: Path to images.jsonl file containing image_id to image_url mappings. If None, uses config.images_jsonl.
        config: Optional BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    # Get paths from config if not provided
    qrels_path = Path(qrels_path) if qrels_path else (Path(config.qrels_jsonl) if config.qrels_jsonl else None)
    output_path = Path(output_path) if output_path else (Path(config.qrels_with_score_jsonl) if config.qrels_with_score_jsonl else None)
    images_jsonl_path = Path(images_jsonl_path) if images_jsonl_path else (Path(config.images_jsonl) if config.images_jsonl else None)
    
    if qrels_path is None:
        raise ValueError("qrels_path must be provided or set in config.qrels_jsonl")
    if output_path is None:
        raise ValueError("output_path must be provided or set in config.qrels_with_score_jsonl")
    if images_jsonl_path is None:
        raise ValueError("images_jsonl_path must be provided or set in config.images_jsonl")
    
    # Determine column name from config or parameter, with fallback
    if col_name is None:
        col_name = config.similarity_config.col_name or "similarity_score"
    
    # Determine adapter name from config or parameter
    if adapter_name is None:
        adapter_name = config.similarity_config.adapter
    
    if not adapter_name:
        available_adapters = SimilarityAdapterRegistry.list_adapters()
        raise ValueError(
            f"Similarity adapter name must be provided or set in config.similarity_config.adapter. "
            f"Available adapters: {', '.join(available_adapters)}"
        )
    
    # Get adapter from registry, passing config
    try:
        adapter = SimilarityAdapterRegistry.get(adapter_name, config=config)
    except Exception as e:
        available_adapters = SimilarityAdapterRegistry.list_adapters()
        raise ValueError(
            f"Failed to initialize similarity adapter '{adapter_name}': {e}. "
            f"Available adapters: {', '.join(available_adapters)}"
        ) from e
    qrels = read_jsonl_list(qrels_path)

    required_columns = config.required_qrels_columns()
    for col in required_columns:
        if not any(col in row for row in qrels):
            raise ValueError(f"Missing required column '{col}' in qrels")

    image_url_map: Dict[str, str] = {}
    if images_jsonl_path and images_jsonl_path.exists():
        images_data = read_jsonl_list(images_jsonl_path)
        for img_row in images_data:
            img_id = img_row.get(config.column_image_id)
            img_url = img_row.get(config.image_url_temp_column)
            if img_id and img_url:
                image_url_map[img_id] = img_url
    else:
        raise ValueError("images_jsonl_path is required for similarity score calculation.")

    total_rows = len(qrels)
    successful = 0
    failed = 0

    # Use tqdm for progress bar
    with tqdm(total=total_rows, desc="Calculating similarity scores", unit="image-query-pair") as pbar:
        for row in qrels:
            try:
                query_text = row.get(config.column_query, "")
                image_id = row.get(config.column_image_id, "")
                if not query_text or not image_id:
                    row[col_name] = None
                    failed += 1
                    pbar.update(1)
                    pbar.set_postfix({"success": successful, "failed": failed})
                    continue

                image_url = image_url_map.get(image_id)
                if not image_url:
                    row[col_name] = None
                    failed += 1
                    pbar.update(1)
                    pbar.set_postfix({"success": successful, "failed": failed})
                    continue

                score = adapter.score(query_text, image_url)
                row[col_name] = score
                successful += 1
                pbar.update(1)
                pbar.set_postfix({"success": successful, "failed": failed})
            except Exception:
                row[col_name] = None
                failed += 1
                pbar.update(1)
                pbar.set_postfix({"success": successful, "failed": failed})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in qrels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def huggingface(
    qrels_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    images_jsonl_path: Optional[Path] = None,
    image_root_dir: Optional[Path] = None,
    progress_interval: int = 100,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    private: Optional[bool] = None,
    config: Optional[BenchmarkConfig] = None,
) -> None:
    """
    Prepare and optionally upload the dataset to Hugging Face Hub.
    
    Args:
        qrels_path: Path to the qrels.jsonl file. If None, uses config.qrels_with_score_jsonl or config.qrels_jsonl.
        output_dir: Directory to save the prepared dataset. If None, uses config.hf_dataset_dir.
        images_jsonl_path: Optional path to images.jsonl file. If None, uses config.images_jsonl.
        image_root_dir: Optional root directory for local images. If None, uses config.image_root_dir.
        progress_interval: Log progress every N rows.
        repo_id: Hugging Face repository ID. If None, uses config._hf_repo_id.
        token: Hugging Face token. If None, uses config._hf_token.
        private: Whether to create a private repository. If None, uses config._hf_private.
        config: Optional BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    # Get paths from config if not provided
    qrels_path = Path(qrels_path) if qrels_path else (Path(config.qrels_with_score_jsonl) if config.qrels_with_score_jsonl else (Path(config.qrels_jsonl) if config.qrels_jsonl else None))
    output_dir = Path(output_dir) if output_dir else (Path(config.hf_dataset_dir) if config.hf_dataset_dir else None)
    images_jsonl_path = Path(images_jsonl_path) if images_jsonl_path else (Path(config.images_jsonl) if config.images_jsonl else None)
    image_root_dir = Path(image_root_dir) if image_root_dir else (Path(config.image_root_dir) if config.image_root_dir else None)
    
    if qrels_path is None:
        raise ValueError("qrels_path must be provided or set in config.qrels_jsonl or config.qrels_with_score_jsonl")
    if output_dir is None:
        raise ValueError("output_dir must be provided or set in config.hf_dataset_dir")
    try:
        from datasets import Dataset, Image as HFImage
    except ImportError as exc:
        raise ImportError("The 'datasets' library is required. Install it with: pip install datasets") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    qrels = read_jsonl_list(qrels_path)
    use_local_images = image_root_dir is not None

    if use_local_images:
        image_root_dir = Path(image_root_dir)
        if not image_root_dir.exists():
            raise ValueError(f"Image root directory does not exist: {image_root_dir}")
    else:
        if not images_jsonl_path or not images_jsonl_path.exists():
            raise ValueError("Either image_root_dir or images_jsonl_path must be provided.")
        images_data = read_jsonl_list(images_jsonl_path)
        image_url_map: Dict[str, str] = {}
        for img_row in images_data:
            img_id = img_row.get(config.column_image_id)
            img_url = img_row.get(config.image_url_temp_column)
            if img_id and img_url:
                image_url_map[img_id] = img_url

    dataset_rows = []
    missing_count = 0
    for idx, row in enumerate(qrels):
        img_id = row.get(config.column_image_id)
        if not img_id:
            missing_count += 1
            continue

        if use_local_images:
            source_path = image_root_dir / img_id
            if not source_path.exists():
                missing_count += 1
                continue
            image_path = str(source_path.resolve())
        else:
            if img_id not in image_url_map:
                missing_count += 1
                continue
            image_path = image_url_map[img_id]

        dataset_row = row.copy()
        dataset_row[config.column_image] = image_path
        dataset_rows.append(dataset_row)

        if (idx + 1) % progress_interval == 0:
            pass

    dataset = Dataset.from_list(dataset_rows)
    dataset = dataset.cast_column("image", HFImage())
    dataset.save_to_disk(str(output_dir / "dataset"))

    metadata = {
        "total_rows": len(dataset_rows),
        "total_qrels": len(qrels),
        "missing_images": missing_count,
        "using_local_images": use_local_images,
        "columns": list(dataset_rows[0].keys()) if dataset_rows else [],
    }
    with open(output_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    repo_id = repo_id or config._hf_repo_id
    token = token or config._hf_token
    if private is None:
        private = config._hf_private if config._hf_private is not None else False

    if repo_id:
        try:
            from huggingface_hub import login, HfApi
        except ImportError as exc:
            raise ImportError("The 'huggingface_hub' library is required. Install it with: pip install huggingface_hub") from exc

        if token:
            login(token=token)
        else:
            api = HfApi()
            api.whoami()

        dataset.push_to_hub(repo_id=repo_id, private=private)


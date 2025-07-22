#!/usr/bin/env python3
"""Plot sentiment analysis results as a labeled scatterplot."""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random


def plot_sentiment_results(sentiment_file: str, output_file: str = None):
    """Create a scatterplot of brand sentiment vs recall.
    
    Args:
        sentiment_file: Path to the sentiment analysis JSON file
        output_file: Optional output filename for the plot
    """
    # Load sentiment data
    with open(sentiment_file, 'r') as f:
        data = json.load(f)
    
    # Extract aggregated metrics
    aggregated = data.get("aggregated_metrics", {})
    brand_metrics = aggregated.get("brand_metrics", {})
    total_analyses = aggregated.get("total_analyses", 1)
    
    # Extract model information from the first response if available
    model_name = "Unknown Model"
    if "individual_results" in data and len(data["individual_results"]) > 0:
        # Try to get model from the response data
        first_result = data["individual_results"][0]
        if "model" in first_result:
            model_name = first_result.get("model", "Unknown Model")
    
    # Prepare data for plotting
    brands = []
    recalls = []
    sentiments = []
    mentions = []
    
    for brand, metrics in brand_metrics.items():
        if metrics["mentions"] >= 3:  # Only plot brands with 3+ mentions
            brands.append(brand)
            recall = (metrics["mentions"] / total_analyses * 100) if total_analyses > 0 else 0
            recalls.append(recall)
            sentiments.append(metrics["net_sentiment"])
            mentions.append(metrics["mentions"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Add random jitter to y-axis for better visibility
    jittered_recalls = []
    for recall in recalls:
        jitter = random.uniform(-0.3, 0.3)
        jittered_recalls.append(recall + jitter)
    
    # Create scatter plot
    scatter = ax.scatter(sentiments, jittered_recalls, 
                        s=[m * 20 for m in mentions],  # Size based on mentions
                        alpha=0.6,
                        c=sentiments,
                        cmap='RdYlGn',
                        vmin=-1, vmax=1,
                        edgecolors='black',
                        linewidth=0.5)
    
    # Add labels for each point
    for i, brand in enumerate(brands):
        # Highlight target brand
        if brand == "Bond":
            ax.annotate(brand, 
                       (sentiments[i], jittered_recalls[i]),
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=10,
                       fontweight='bold',
                       color='darkblue',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        else:
            ax.annotate(brand, 
                       (sentiments[i], jittered_recalls[i]),
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.8)
    
    # Add quadrant lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Add quadrant backgrounds
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Positive sentiment, high recall (top right) - light green
    ax.add_patch(Rectangle((0, 0), xlim[1], ylim[1], 
                          facecolor='green', alpha=0.05))
    
    # Negative sentiment, high recall (top left) - light red
    ax.add_patch(Rectangle((xlim[0], 0), -xlim[0], ylim[1], 
                          facecolor='red', alpha=0.05))
    
    # Set labels and title
    ax.set_xlabel('Average Sentiment Score', fontsize=12)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Brand Sentiment vs Recall Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.text(0.5, 0.98, f'Response Model: {model_name}', 
            transform=fig.transFigure, 
            fontsize=12, 
            ha='center', 
            va='top',
            color='gray')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1, max(jittered_recalls) + 2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sentiment Score', fontsize=10)
    
    # Add legend for size
    sizes = [3, 10, 20]
    labels = ['3 mentions', '10 mentions', '20 mentions']
    legend_elements = []
    for size, label in zip(sizes, labels):
        legend_elements.append(plt.scatter([], [], s=size*20, c='gray', alpha=0.6, 
                                         edgecolors='black', linewidth=0.5, label=label))
    
    legend1 = ax.legend(handles=legend_elements, loc='lower right', 
                       title='Number of Mentions', framealpha=0.9)
    ax.add_artist(legend1)
    
    # Add summary statistics
    stats_text = f"Total Brands: {len(brands)}\nTotal Responses Analyzed: {total_analyses}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        # Default filename based on input
        output_file = sentiment_file.replace('.json', '_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.close()
    
    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_sentiment_results.py <sentiment_json_file> [output_file]")
        sys.exit(1)
    
    sentiment_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_sentiment_results(sentiment_file, output_file)


if __name__ == "__main__":
    main()
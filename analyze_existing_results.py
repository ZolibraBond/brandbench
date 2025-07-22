#!/usr/bin/env python3
"""Analyze sentiment in existing test results."""

import json
import sys
from datetime import datetime
from src.evaluation.sentiment_analyzer import SentimentAnalyzer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def analyze_existing_results(results_file: str, max_items: int = None):
    """Analyze sentiment in existing test results."""
    console = Console()
    
    # Load existing results
    console.print(f"[bold blue]Loading results from {results_file}...[/bold blue]")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Handle both old format and new format
    if "responses" in data:
        # New format with metadata
        test_results = data["responses"]
        metadata = data.get("metadata", {})
    else:
        # Old format - direct responses
        test_results = data
        metadata = {}
    
    # Initialize the analyzer
    console.print("[bold blue]Initializing Sentiment Analyzer...[/bold blue]")
    analyzer = SentimentAnalyzer()
    
    # Collect responses to analyze
    responses_to_analyze = []
    count = 0
    
    for category, results in test_results.items():
        for result in results:
            if "content" in result and result["content"]:
                responses_to_analyze.append({
                    "category": category,
                    "prompt": result["prompt"],
                    "content": result["content"],
                    "web_search_enabled": result.get("web_search_enabled", False)
                })
                count += 1
                if max_items and count >= max_items:
                    break
        if max_items and count >= max_items:
            break
    
    # Analyze each response
    console.print(f"\n[bold yellow]Analyzing {len(responses_to_analyze)} responses...[/bold yellow]\n")
    
    all_results = []
    
    for i, response in enumerate(responses_to_analyze):
        console.print(f"\n[{i+1}/{len(responses_to_analyze)}] ", end="")
        console.print(f"[cyan]{response['category']}[/cyan] - {response['prompt'][:60]}...")
        
        # Analyze sentiment
        sentiments = analyzer.analyze_sentiment(response['content'])
        
        # Show all brands with their sentiments
        mentioned_brands = []
        for brand, score in sentiments.items():
            if score is not None:
                mentioned_brands.append((brand, score))
        
        if mentioned_brands:
            # Sort by sentiment (positive first) then by brand name
            mentioned_brands.sort(key=lambda x: (-x[1], x[0]))
            brand_strs = []
            for brand, score in mentioned_brands:
                sentiment_text = {1: "+", 0: "=", -1: "-"}.get(score, "?")
                color = {1: "green", 0: "yellow", -1: "red"}.get(score, "white")
                brand_strs.append(f"[{color}]{brand}({sentiment_text})[/{color}]")
            console.print("Brands: " + ", ".join(brand_strs))
        else:
            console.print("[yellow]No brands mentioned[/yellow]")
        
        # Save result
        analysis_result = {
            "prompt": response['prompt'],
            "category": response['category'],
            "web_search_enabled": response['web_search_enabled'],
            "content": response['content'],
            "sentiments": sentiments,
            "mentioned_brands": [b[0] for b in mentioned_brands]
        }
        all_results.append(analysis_result)
    
    # Aggregate and display summary
    console.print("\n[bold yellow]Aggregating results...[/bold yellow]\n")
    
    aggregated = analyzer.aggregate_sentiments(all_results)
    
    # Display aggregated metrics
    summary_table = Table(show_header=True, header_style="bold magenta", title="Brand Sentiment Summary")
    summary_table.add_column("Brand", style="cyan")
    summary_table.add_column("Recall", justify="center")
    summary_table.add_column("Average Sentiment", justify="center")
    summary_table.add_column("Mentions", justify="center", style="dim")
    
    # Calculate recall percentage for each brand
    total_prompts = aggregated["total_analyses"]
    
    # Sort by recall (mentions/total_prompts), then by brand name
    brands_with_recall = []
    for brand, metrics in aggregated["brand_metrics"].items():
        recall = (metrics["mentions"] / total_prompts * 100) if total_prompts > 0 else 0
        brands_with_recall.append((brand, recall, metrics))
    
    sorted_brands = sorted(brands_with_recall, key=lambda x: (-x[1], x[0]))
    
    for brand, recall, metrics in sorted_brands:
        if metrics["mentions"] > 0:
            # Calculate average sentiment (-1 to 1 scale)
            avg_sentiment = metrics["net_sentiment"]
            
            # Color code based on sentiment
            if avg_sentiment > 0.5:
                sentiment_color = "green"
            elif avg_sentiment > 0:
                sentiment_color = "yellow" 
            elif avg_sentiment >= -0.5:
                sentiment_color = "orange"
            else:
                sentiment_color = "red"
            
            summary_table.add_row(
                brand,
                f"{recall:.1f}%",
                f"[{sentiment_color}]{avg_sentiment:+.2f}[/{sentiment_color}]",
                str(metrics["mentions"])
            )
    
    console.print(summary_table)
    
    # Show target brand specifically
    target_brand = "Bond"
    if target_brand in aggregated["brand_metrics"]:
        target_metrics = aggregated["brand_metrics"][target_brand]
        target_recall = (target_metrics["mentions"] / total_prompts * 100) if total_prompts > 0 else 0
        console.print(f"\n[bold cyan]{target_brand} (Target Brand):[/bold cyan]")
        console.print(f"  Recall: {target_recall:.1f}%")
        console.print(f"  Average Sentiment: {target_metrics['net_sentiment']:+.2f}")
        console.print(f"  Total Mentions: {target_metrics['mentions']}")
    
    # Save results with proper structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save sentiment analysis
    sentiment_file = f"data/sentiment/{timestamp}_sentiment.json"
    sentiment_data = {
        "timestamp": timestamp,
        "source_responses": results_file,
        "metadata": metadata,
        "individual_results": all_results,
        "aggregated_metrics": aggregated
    }
    
    with open(sentiment_file, 'w') as f:
        json.dump(sentiment_data, f, indent=2)
    
    console.print(f"\n[green]Sentiment analysis saved to {sentiment_file}[/green]")
    
    # Save analysis report
    report_file = f"data/analysis/{timestamp}_analysis.json"
    report_data = {
        "timestamp": timestamp,
        "files": {
            "responses": results_file,
            "sentiment": sentiment_file
        },
        "summary": {
            "total_prompts_analyzed": len(all_results),
            "brands_mentioned": len([b for b, m in aggregated["brand_metrics"].items() if m["mentions"] > 0]),
            "target_brand": "Bond",
            "target_brand_metrics": aggregated["brand_metrics"].get("Bond", {}),
            "top_5_brands": [
                {"brand": brand, "recall": f"{recall:.1f}%", "avg_sentiment": metrics["net_sentiment"], "mentions": metrics["mentions"]}
                for brand, recall, metrics in sorted_brands[:5]
                if metrics["mentions"] > 0
            ]
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    console.print(f"[green]Analysis report saved to {report_file}[/green]")


def main():
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Default to the most recent results file
        results_file = "test_results_20250722_103221.json"
    
    max_items = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    analyze_existing_results(results_file, max_items)


if __name__ == "__main__":
    main()
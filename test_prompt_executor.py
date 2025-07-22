#!/usr/bin/env python3
"""Test script for the prompt executor."""

import argparse
import json
from datetime import datetime
from src.benchmark.prompt_executor import PromptExecutor
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def test_prompt_executor(
    prompt_file: str,
    num_prompts: int = 5,
    enable_web_search: bool = False,
    category: str = None
):
    """Test the prompt executor with a limited number of prompts."""
    console = Console()
    
    # Initialize the executor
    console.print("[bold blue]Initializing Prompt Executor...[/bold blue]")
    executor = PromptExecutor()
    
    # Load prompts
    prompts_by_category = executor.load_prompts(prompt_file)
    
    # Filter by category if specified
    if category:
        if category in prompts_by_category:
            prompts_by_category = {category: prompts_by_category[category]}
        else:
            console.print(f"[red]Category '{category}' not found in prompt file[/red]")
            return
    
    # Limit prompts per category
    limited_prompts = {}
    for cat, prompts in prompts_by_category.items():
        limited_prompts[cat] = prompts[:num_prompts]
    
    # Display what we're testing
    total_prompts = sum(len(p) for p in limited_prompts.values())
    console.print(f"\n[bold]Testing {total_prompts} prompts from {len(limited_prompts)} categories[/bold]")
    console.print(f"Web Search: [{'green' if enable_web_search else 'red'}]{'Enabled' if enable_web_search else 'Disabled'}[/]")
    
    # Execute prompts
    all_results = {}
    for cat, prompts in limited_prompts.items():
        console.print(f"\n[bold yellow]Processing category: {cat}[/bold yellow]")
        
        results = executor.execute_batch(
            prompts, 
            enable_web_search=enable_web_search,
            batch_size=5
        )
        all_results[cat] = results
        
        # Display results in a table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Prompt", style="cyan", max_width=50)
        table.add_column("Response Preview", style="white", max_width=80)
        table.add_column("Time (s)", style="green")
        
        if enable_web_search:
            table.add_column("URLs", style="blue", max_width=40)
        
        for result in results:
            prompt = result.get("prompt", "")[:50] + "..." if len(result.get("prompt", "")) > 50 else result.get("prompt", "")
            
            if "error" in result:
                table.add_row(
                    prompt,
                    f"[red]Error: {result['error'][:80]}[/red]",
                    "-",
                    "-" if enable_web_search else None
                )
            else:
                content = result.get("content", "")[:80] + "..." if len(result.get("content", "")) > 80 else result.get("content", "")
                response_time = f"{result.get('response_time', 0):.2f}"
                
                if enable_web_search:
                    urls = result.get("cited_urls", [])
                    url_text = f"{len(urls)} URLs" if urls else "No URLs"
                    table.add_row(prompt, content, response_time, url_text)
                else:
                    table.add_row(prompt, content, response_time)
        
        console.print(table)
    
    # Save results with proper structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/responses/{timestamp}_responses.json"
    
    # Prepare data with metadata
    data = {
        "timestamp": timestamp,
        "metadata": {
            "prompt_file": prompt_file,
            "web_search_enabled": enable_web_search,
            "categories": list(all_results.keys()),
            "total_prompts": sum(len(results) for results in all_results.values())
        },
        "responses": all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    console.print(f"\n[bold green]Results saved to {output_file}[/bold green]")
    
    # Display summary statistics
    console.print("\n[bold]Summary Statistics:[/bold]")
    total_responses = sum(len(results) for results in all_results.values())
    total_errors = sum(1 for results in all_results.values() for r in results if "error" in r)
    avg_time = sum(r.get("response_time", 0) for results in all_results.values() for r in results if "response_time" in r) / (total_responses - total_errors) if total_responses > total_errors else 0
    
    stats_table = Table(show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row("Total Prompts", str(total_responses))
    stats_table.add_row("Successful", str(total_responses - total_errors))
    stats_table.add_row("Errors", str(total_errors))
    stats_table.add_row("Avg Response Time", f"{avg_time:.2f}s")
    
    if enable_web_search:
        total_with_urls = sum(1 for results in all_results.values() for r in results if r.get("cited_urls"))
        stats_table.add_row("Responses with URLs", str(total_with_urls))
    
    console.print(stats_table)


def main():
    parser = argparse.ArgumentParser(description="Test the BrandBench prompt executor")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompts/research_intent.yaml",
        help="Path to the prompt YAML file"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to test per category"
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Enable web search for prompts"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Test only a specific category"
    )
    
    args = parser.parse_args()
    
    test_prompt_executor(
        args.prompt_file,
        args.num_prompts,
        args.web_search,
        args.category
    )


if __name__ == "__main__":
    main()
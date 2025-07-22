# BrandBench

Tracking brand knowledge and sentiment in AI chatbots.

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Running BrandBench

#### 1. Execute Prompts

Test prompts without web search:
```bash
python test_prompt_executor.py
```

Test prompts with web search enabled:
```bash
python test_prompt_executor.py --web-search
```

Options:
- `--prompt-file`: Path to prompt file (default: `prompts/prompts.yaml`)
- `--num-prompts`: Total number of prompts to test (default: all)
- `--category`: Test only a specific category (e.g., `ceiling_fans`, `shades`)
- `--web-search`: Enable web search for responses

Examples:
```bash
# Test all prompts with web search
python test_prompt_executor.py --web-search

# Test only 10 prompts total
python test_prompt_executor.py --num-prompts 10

# Test only shades category
python test_prompt_executor.py --category shades --web-search
```

#### 2. Analyze Sentiment

Analyze the most recent results:
```bash
python analyze_existing_results.py
```

Analyze a specific results file:
```bash
python analyze_existing_results.py data/responses/20250722_143021_responses.json
```

### Output

- **Responses**: Saved to `data/responses/YYYYMMDD_HHMMSS_responses.json`
- **Sentiment Analysis**: Saved to `data/sentiment/YYYYMMDD_HHMMSS_sentiment.json`
- **Analysis Reports**: Saved to `data/analysis/YYYYMMDD_HHMMSS_analysis.json`

### Understanding Results

The sentiment analysis shows:
- **Recall**: Percentage of prompts where each brand was mentioned
- **Average Sentiment**: Average sentiment score (-1 to +1) when mentioned
  - +1: Positive (recommended, praised)
  - 0: Neutral (factual mention)
  - -1: Negative (criticized, problematic)

## Business Purpose

Consumers are increasingly using chatbots in their purchasing journey.  ChatGPT is being used to discover solutions and products as well as to find product information pre- and post-sale.  It is in Olibra's interest to understand---and to the extent possible---to influence the information that chatbots provide to consumers so as to increase sales and ensure our brand is properly represented.

## Objective

At a high level, we seek to measure, on a recurring basis, the following:

 1. **Recall** -- How often does the chatbot suggest Bond relative to competing solutions?
 2. **Sentiment** -- Is the Bond brand presented in a positive, negative, or neutral way?
 3. **Accuracy** -- Is the information provided on Bond products accurate?

These metrics will allow us to measure any interventions we undertake, and we expect the details of the project to give us insight into what interventions will be most effective (for example, by discovering the specific search terms that the chatbots are using but have low brand recall).

For now, we focus solely on recall and sentiment. Accuracy is a broader subject to be taken up at a later date.

## Methodology

We simulate consumer interaction with chatbots by leveraging the OpenAI API access to the GPT-4o model, which very closely approximates results obtained when using ChatGPT.

We focus entirely on OpenAI's system as we expect the vast majority of consumer-chatbot interaction to be occurring on the ChatGPT product.

We construct a battery of short prompts that are typical of what prospective customers may type into ChatGPT and then use other AI models to judge the responses.

### Prompts

We test several categories of prompts:

#### Research Intent

Prompts where users are trying to solve a home automation problem:

 - How can I control my shades with Alexa?
 - How can I control Somfy shades?
 - Cloning RF remote controls.

#### Shopping Intent

Prompts that indicate a stronger purchase intent:

 - Compare prices on smart home hubs for shade control.
 - Best smart ceiling fans.
 - Are there alternatives to TaHoma?

#### Product Queries

In the future we may use FAQ-style questions to test accuracy of chatbot knowledge.  This will be focused on pre-sales questions that have an impact on the probability of purchase, however in the future we may wish to assess the chatbot's responses for unrealistic expectation setting and technical support post-sale.

 - Does Bond Bridge work with SAVANT?
 - Can Bond Bridge control Lutron shades?
 - Does Bond Bridge connect with Apple HomeKit?

### Intrinsic Knowledge vs Web Search

We will study both the intrinsic behavior of the model as well as the performance when a web search is performed.  We will focus more on the prompts that tend to generate web search tool calls as these results can be more readily intervened upon. Changing models' intrinsic knowledge requires a longer-term campaign of high-quality outbound content to get into the pre-training data. This has a 6--12 mo leadtime for current frontier models.

## Outcomes

This project will deliver:

 - graphs showing brand performance over time
 - specific search terms used by chatbots, highlighting those where Bond has poor performance
 - raw data available for re-processing at a later date
 









import contextlib
import pandas as pd
from openai import OpenAI
import re
from tqdm import tqdm

tqdm.pandas(desc="Processing data")

client = OpenAI(api_key="sk-c1b90a0a26b5423f9097a600499e328e", base_url="https://api.deepseek.com")

def evaluate_context(row):
    """
    Generate contextual reasoning and confidence using DeepSeek API.
    """
    # Construct the prompt
    prompt = f"""
    You are a financial analyst assisting a trading agent. Based on the following data:
    - Ask volume: {row['ask']}
    - Bid volume: {row['bid']}
    - SMA comparison: {row['sma-compare']} (a boolean value indicating whether sma-7 > sma-5 and sma-25 > sma-7)
    - Current position: {'short' if row['is_short'] else 'long'}

    Provide a detailed analysis of the current market context and suggest whether the agent should go_long, go_short, or do_nothing. 
    Include your reasoning and MUST assign a confidence score (0 to 1) depending on whether the market is BEARISH or BULLISH and the current position.

    **Response Format:**
    - Reasoning: <your reasoning here>
    - Confidence: <confidence score as a float between 0 and 1>
    """

    # Call the DeepSeek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
        stream=False
    )

    # Extract the reasoning from the response
    reasoning = response.choices[0].message.content.strip()

    # Parse the confidence score (assuming the LLM outputs it in a specific format)
    confidence = 0.5  # Default confidence
    confidence_match = re.search(r"confidence: (\d\.\d+)", reasoning)

    if "confidence:" in reasoning:
        with contextlib.suppress(Exception):
            confidence = float(confidence_match[1]) # float(reasoning.split("confidence:")[1].split()[0])

    return reasoning, confidence

df = pd.read_csv('/home/defi/Desktop/portfolio/projects/python/pipeline_defi/lean_df.csv')

#df = df.head(10)
# Add contextual reasoning and confidence to the DataFrame
df[['reasoning', 'confidence']] = df.progress_apply(evaluate_context, axis=1, result_type='expand')

# Display the updated DataFrame
select_ = ['reasoning', 'confidence']
print(df['confidence'].head())
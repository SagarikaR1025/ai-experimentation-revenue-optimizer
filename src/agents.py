from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def generate_campaign_recommendation(summary_metrics):

    prompt = f"""
    You are a growth analytics consultant.

    Analyze the following campaign performance metrics and provide executive recommendations.

    Focus on:
    - optimization opportunities
    - budget allocation
    - conversion improvement
    - customer acquisition efficiency
    - campaign scaling recommendations

    Campaign Metrics:
    {summary_metrics}

    Provide:
    - executive summary
    - optimization recommendations
    - risk considerations
    - next best actions

    Keep the response concise and professional.
    """

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text
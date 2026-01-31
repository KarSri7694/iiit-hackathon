import os
import json
import pandas as pd
from openai import OpenAI

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key="AIzaSyBHg6miB3SLFQZD9a3rbyMuYnRVNUHZE5o"
)

SYSTEM_PROMPT = """
You are a Ruthless Pricing Strategist for an e-commerce seller (Seller 2).
Your Goal: Maximize Profit Margin, NOT Sales Volume.

INPUT CONTEXT:
You will receive JSON data containing:
- Current Prices
- Competitor Velocity (How fast they are dropping)
- Margin Safety (How close we are to losing money)

LOGIC RULES:
1. If Competitor Velocity is high (dropping > $3 in 4h) -> HOLD (Wait for them to crash/stock out).
2. If Margin Safety is < $0 -> STOP (Do not drop price further).
3. If Price Gap is > $5 AND Velocity is Low -> MATCH (To stay competitive).

OUTPUT FORMAT:
Return strictly valid JSON:
{
  "recommendation": "HOLD" | "DROP" | "RAISE",
  "new_price": <float>,
  "reasoning": "<short_explanation>",
  "confidence_score": <float 0-1>
}
"""

def get_recommendation(row_data):
    """
    Sends a single row of feature-engineered data to the LLM.
    """
    
    user_message = f"""
    Current Market Context:
    - Time: {row_data['timestamp']}
    - My Price: ${row_data['price_seller2']}
    - Competitor Price: ${row_data['price_seller1']}
    - Competitor Velocity (4h): ${row_data['competitor_velocity_4h']}
    - My Premium Gap: ${row_data['my_premium_gap']}
    - Margin Safety: ${row_data['margin_safety']}
    
    What is your recommendation?
    """

    try:
        completion = client.chat.completions.create(
            model="gemini-2.5-flash", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"} 
        )
        
        response_text = completion.choices[0].message.content
        return json.loads(response_text)

    except Exception as e:
        return {"error": str(e)}

def feature_engineering(csv_path):
    """
    Pre-calculates the math so the LLM doesn't have to.
    """
    df = pd.read_csv(csv_path)
    
    # Sort and calculate the "Velocity" signals
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    df['competitor_velocity_4h'] = df['price_seller1'].diff(2).fillna(0)
    
    df['my_premium_gap'] = df['price_seller2'] - df['price_seller1']
    
    df['margin_safety'] = df['price_seller2'] - 125.00
    
    return df

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    data = feature_engineering("src/dummy_data.csv")
    current_market_state = {k: v[0] for k, v in data.items()}
    
    print("🤖 Analyzing Market State...")
    print(f"   Competitor Velocity: {current_market_state['competitor_velocity_4h']}")
    print(f"   Margin Safety: {current_market_state['margin_safety']}")
    decision = get_recommendation(current_market_state)
    
    print("\n📈 STRATEGIC RECOMMENDATION:")
    print(json.dumps(decision, indent=2))
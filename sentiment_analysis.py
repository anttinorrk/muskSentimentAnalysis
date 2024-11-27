import os
import pandas as pd
from openai import OpenAI

df = pd.read_csv("analyzed_relatedness.csv") #Change the path
data_sample = df #.iloc[0:10].copy() #Uncomment to use a small data sample


api_key = [API KEY]

sentiment_prompt = """
Analyze the sentiment of the following tweet to Neutral, Positive, or Negative, considering the tone of the tweet.

If the tweet is neither particularly optimistic or pessimistic or has mainly an informative vibe, it should be neutral.
Also, if you're not confident about the sentiment, you should gravitate towards more neutral sentiment.

The sentiment should be given in a numeric format, where -1 means strong negative, 0 means neutral, and 1 means strong positive. 
Values close to 0 represent milder sentiment.
Response should contain only the numerical representation (for example -0.4 or 0.8) and nothing else.
"""

relatedness_prompt = """
Analyze how much the following tweet is related to Tesla.
Consider tweets about Tesla's products, business, stock, performance etc. as well as tweets related to Tesla's competition and market outlook of the electric car industry.

Consider in the analysis that Elon Musk has other companies like SpaceX that operates in space shuttle business. 
Relatedness to SpaceX or space shuttles should not be considered as related to Tesla.

Rate relatedness in a numeric format between 0 and 1.
0 means that tweet is not related to Tesla at all, and 1 that it is related to Tesla.
If the tweet is somewhat relevant to Tesla but not directly about it, rate it somewhere in the middle.

Response should contain only the numeric value (for example 0, 0.4, or 1) and nothing else.
"""

opinion_prompt = """
Analyze the strength of personal opinion of this tweet. Consider factors such as how subjective the tweet is and how strong emotional intensity it contains and how assertive it is.

Rate the strength of the personal opinion in a numeric format between 0 and 1.

If the tweet is purely informational and contains no subjectivity or emotions, it should be rated as 0.
If the tweet contains strong personal opinion, emotions, or assertiveness, it should be 1.
If there is some level of personal opinion or emotions, it should be rated somewhere in the middle.

Response should contain only the numeric value (for example 0, 0.3, or 1) and nothing else.
"""

# Initialize OpenAI connection
client = OpenAI(api_key=api_key)

def get_sentiment(text):
    try:
        response=client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant specialized in analyzing tweets from electric car manufacturer Tesla's CEO Elon Musk."},
                {"role": "user", "content": f"{sentiment_prompt}:\n\n{text}"} #Change used prompt
            ],
            max_tokens=200,
            temperature=0  # Lower temperature for more predictable results
        )
        print(response)
        sentiment = response.choices[0].message.content.strip()
        return sentiment
    except Exception as e:
        print(f"Error processing row: {e}")
        return None
 
    
data_sample['Sentiment'] = data_sample['Tweet'].apply(get_sentiment) #Change field name to create

data_sample.to_csv("analyzed_relatedness_sentiment.csv")

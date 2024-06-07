import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    labels = ["Negative", "Neutral", "Positive"]
    sentiment_label = labels[scores.argmax()]
    return sentiment_label

user_input = input("Enter your text for sentiment analysis: ")
sentiment = analyze_sentiment(user_input)
print(f"'{user_input}' - Sentiment: {sentiment}")

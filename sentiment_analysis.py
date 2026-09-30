import pandas as pd
from textblob import TextBlob
import re

# Load 5,000 customer reviews
df = pd.read_csv("reviews.csv", nrows=5000)

# Keep only the review text
df = df[["body"]]

# Remove empty reviews
df = df.dropna(subset=["body"])

# Remove duplicate reviews
df = df.drop_duplicates()

print("Reviews loaded successfully!")
print("Number of reviews:", len(df))


# ---------------- SENTIMENT ANALYSIS ----------------

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


# Apply sentiment analysis
df["sentiment"] = df["body"].apply(get_sentiment)

print("\nSentiment analysis completed!")

print("\nFirst 10 sentiment results:")
print(df[["body", "sentiment"]].head(10))


# Count sentiments
sentiment_counts = df["sentiment"].value_counts()

print("\nOverall Sentiment Distribution:")
print(sentiment_counts)


# ---------------- EMOTION ANALYSIS ----------------

# Emotion word lists
emotion_words = {
    "Joy": [
        "happy", "happiness", "joy", "love", "excellent",
        "amazing", "wonderful", "great", "delight", "excited"
    ],

    "Trust": [
        "trust", "trusted", "reliable", "safe", "secure",
        "honest", "quality", "dependable"
    ],

    "Fear": [
        "fear", "afraid", "scared", "worried", "worry",
        "danger", "dangerous", "anxious"
    ],

    "Anger": [
        "angry", "anger", "hate", "annoyed", "annoying",
        "frustrated", "frustrating", "terrible"
    ],

    "Sadness": [
        "sad", "sadness", "unhappy", "disappointed",
        "disappointing", "cry", "hurt", "poor"
    ],

    "Surprise": [
        "surprise", "surprised", "unexpected",
        "shocked", "amazed", "astonished"
    ],

    "Anticipation": [
        "hope", "hopeful", "expect", "expected",
        "waiting", "future", "excited"
    ],

    "Disgust": [
        "disgust", "disgusting", "awful", "gross",
        "horrible", "worst", "dirty"
    ]
}


def detect_emotions(text):
    text = str(text).lower()
    words = re.findall(r"\b\w+\b", text)

    detected = []

    for emotion, keywords in emotion_words.items():
        for word in keywords:
            if word in words:
                detected.append(emotion)
                break

    if len(detected) == 0:
        return "No specific emotion detected"

    return ", ".join(detected)


# Apply emotion analysis
df["emotions"] = df["body"].apply(detect_emotions)

print("\nEmotion analysis completed!")

print("\nFirst 10 emotion results:")
print(df[["body", "emotions"]].head(10))


# Count emotions
emotion_counts = {}

for emotions in df["emotions"]:
    for emotion in emotions.split(", "):
        if emotion != "No specific emotion detected":
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

print("\nEmotion Distribution:")

for emotion, count in sorted(
    emotion_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(emotion, ":", count)


# ---------------- SAVE RESULTS ----------------

df.to_csv("sentiment_emotion_results.csv", index=False)

print("\nAnalysis completed successfully!")
print("Results saved as: sentiment_emotion_results.csv")
import matplotlib.pyplot as plt

# Create emotion distribution chart
plt.figure(figsize=(10, 6))

plt.bar(emotion_counts.keys(), emotion_counts.values())

plt.title("Emotion Distribution in Customer Reviews")
plt.xlabel("Emotions")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

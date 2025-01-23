# Sentiment Agent
class SentimentAgent:
    """
    Analyzes the sentiment of a given text.

    Methods:
        analyze_sentiment(text: str) -> float:
            Analyzes the sentiment of the input text and returns a polarity score.
            Parameters:
                text (str): The text to analyze.
            Returns:
                float: Sentiment polarity score ranging from -1.0 (negative) to 1.0 (positive).
    """
    def analyze_sentiment(self, text):
        analysis = text
        return analysis.sentiment.polarity
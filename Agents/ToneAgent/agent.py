# Tone Agent
class ToneAgent:
    """
    Adjusts the tone of aggregated data based on sentiment score.

    Methods:
        adjust_tone_with_persona(data: str, sentiment_score: float) -> str:
            Adjusts the tone of the given data according to the sentiment score.
            Parameters:
                data (str): Aggregated data from the database.
                sentiment_score (float): Sentiment polarity score.
            Returns:
                str: Tone-adjusted data.
    """
    def adjust_tone_with_persona(self, data, sentiment_score):
        if sentiment_score > 0:
            return f"Positive tone: {data}"
        elif sentiment_score == 0:
            return f"Neutral tone: {data}"
        else:
            return f"Negative tone: {data}"
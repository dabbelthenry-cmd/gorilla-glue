import datetime

class NewsService:
    def __init__(self):
        # Mock news data for demonstration
        # In production, this would connect to NewsAPI, Bloomberg, or Reuters
        self.NEWS_DATA = {
            "USD": [
                {"title": "Fed Signals Potential Rate Cut in Late 2025", "source": "Financial Times", "time": "2 hours ago"},
                {"title": "US GDP Growth Exceeds Expectations in Q3", "source": "Bloomberg", "time": "5 hours ago"},
                {"title": "Treasury Yields Stabilize Ahead of Jobs Report", "source": "Reuters", "time": "8 hours ago"}
            ],
            "EUR": [
                {"title": "ECB President Warns of Persisting Inflation Risks", "source": "Euronews", "time": "1 hour ago"},
                {"title": "German Manufacturing PMI Shows Signs of Recovery", "source": "CNBC Europe", "time": "4 hours ago"},
                {"title": "Eurozone Consumer Confidence Improves Slightly", "source": "MarketWatch", "time": "6 hours ago"}
            ],
            "GBP": [
                {"title": "Bank of England Holds Rates Steady at 5.25%", "source": "BBC Business", "time": "30 mins ago"},
                {"title": "UK Inflation Falls to Lowest Level in Two Years", "source": "The Guardian", "time": "3 hours ago"},
                {"title": "Sterling Rallies Against Dollar on Strong Retail Sales", "source": "Reuters", "time": "7 hours ago"}
            ],
            "JPY": [
                {"title": "BOJ Governor Hints at Policy Normalization", "source": "Nikkei Asia", "time": "2 hours ago"},
                {"title": "Yen Weakens as Trade Deficit Widens", "source": "Japan Times", "time": "5 hours ago"},
                {"title": "Japan's Core Inflation Stays Above 2% Target", "source": "Bloomberg", "time": "9 hours ago"}
            ],
            "CHF": [
                {"title": "SNB Surprise Rate Cut Boosts Swiss Stocks", "source": "Swissinfo", "time": "4 hours ago"},
                {"title": "Swiss Franc Safe-Haven Appeal Diminishes", "source": "Reuters", "time": "10 hours ago"}
            ],
            "AUD": [
                {"title": "RBA Minutes Reveal Hawkish Stance on Inflation", "source": "ABC News", "time": "3 hours ago"},
                {"title": "Australian Dollar Dips on Weak Commodity Prices", "source": "Financial Review", "time": "6 hours ago"}
            ]
        }

    def get_news(self, currency):
        """
        Returns a list of news items for the given currency.
        """
        return self.NEWS_DATA.get(currency, [])

    def get_combined_news(self, base_currency, quote_currency):
        """
        Returns combined news for the currency pair.
        """
        base_news = self.get_news(base_currency)
        quote_news = self.get_news(quote_currency)
        
        # Combine and "sort" (interleave for demo)
        combined = []
        len_base = len(base_news)
        len_quote = len(quote_news)
        max_len = max(len_base, len_quote)
        
        for i in range(max_len):
            if i < len_base:
                item = base_news[i].copy()
                item['currency'] = base_currency
                combined.append(item)
            if i < len_quote:
                item = quote_news[i].copy()
                item['currency'] = quote_currency
                combined.append(item)
                
        return combined

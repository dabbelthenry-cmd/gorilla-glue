import pandas as pd

class FundamentalsService:
    def __init__(self):
        # Snapshot of recent economic data (approximate values for Demo)
        # In a production app, this would fetch from FRED or TradingEconomics API
        self.ECONOMIC_DATA = {
            "USD": {
                "Region": "United States",
                "GDP Growth (YoY)": "2.9%",
                "Unemployment Rate": "3.7%",
                "CPI (Inflation)": "3.4%",
                "PPI (YoY)": "0.9%",
                "Interest Rate": "5.50%"
            },
            "EUR": {
                "Region": "Eurozone",
                "GDP Growth (YoY)": "0.1%",
                "Unemployment Rate": "6.4%",
                "CPI (Inflation)": "2.9%",
                "PPI (YoY)": "-8.6%",
                "Interest Rate": "4.50%"
            },
            "GBP": {
                "Region": "United Kingdom",
                "GDP Growth (YoY)": "0.3%",
                "Unemployment Rate": "4.2%",
                "CPI (Inflation)": "4.0%",
                "PPI (YoY)": "-2.6%",
                "Interest Rate": "5.25%"
            },
            "JPY": {
                "Region": "Japan",
                "GDP Growth (YoY)": "1.2%",
                "Unemployment Rate": "2.5%",
                "CPI (Inflation)": "2.6%",
                "PPI (YoY)": "0.0%",
                "Interest Rate": "-0.10%"
            },
            "CHF": {
                "Region": "Switzerland",
                "GDP Growth (YoY)": "0.7%",
                "Unemployment Rate": "2.0%",
                "CPI (Inflation)": "1.7%",
                "PPI (YoY)": "-0.9%",
                "Interest Rate": "1.75%"
            },
            "AUD": {
                "Region": "Australia",
                "GDP Growth (YoY)": "2.1%",
                "Unemployment Rate": "3.9%",
                "CPI (Inflation)": "4.1%",
                "PPI (YoY)": "4.1%",
                "Interest Rate": "4.35%"
            }
        }

    def get_fundamentals(self, currency):
        """
        Returns a dictionary of fundamental data for the given currency code.
        """
        # Handle potential variations or defaults
        return self.ECONOMIC_DATA.get(currency, {})

    def get_comparison_df(self, base_currency, quote_currency):
        """
        Returns a DataFrame comparing the two currencies.
        """
        base_data = self.get_fundamentals(base_currency)
        quote_data = self.get_fundamentals(quote_currency)
        
        if not base_data or not quote_data:
            return None

        metrics = ["Region", "GDP Growth (YoY)", "Unemployment Rate", "CPI (Inflation)", "PPI (YoY)", "Interest Rate"]
        
        data = {
            "Metric": metrics,
            base_currency: [base_data.get(m, "N/A") for m in metrics],
            quote_currency: [quote_data.get(m, "N/A") for m in metrics]
        }
        
        return pd.DataFrame(data)

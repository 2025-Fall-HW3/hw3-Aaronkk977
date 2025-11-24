"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # Sector Momentum / Rotation Strategy
        # Logic: Buy top N performing sectors over lookback period
        # Market Filter: Reduce exposure if SPY is in downtrend (MA slope < 0)
        
        top_n = 3
        ma_window = 20  # Monthly Moving Average for trend filter
        
        # Calculate SPY trend (using the excluded asset column if it is SPY, or get it from price)
        # Assuming 'SPY' is in self.price or self.exclude is 'SPY'
        spy_price = self.price[self.exclude] if self.exclude in self.price.columns else None
        
        for i in range(self.lookback + 1, len(self.price)):
            current_date = self.price.index[i]
            
            # 1. Calculate Momentum (Cumulative Return over lookback)
            # Get prices for lookback window
            past_prices = self.price[assets].iloc[i - self.lookback : i]
            current_prices = self.price[assets].iloc[i]
            
            # Momentum = (Current Price / Price lookback days ago) - 1
            momentum = (current_prices / past_prices.iloc[0]) - 1
            
            # 2. Rank and Select Top N
            top_assets = momentum.nlargest(top_n).index
            
            # 3. Market Trend Filter (Moving Average Slope)
            market_condition = 1.0 # Default to full investment
            
            if spy_price is not None and i > ma_window:
                # Calculate MA over ma_window
                spy_history = spy_price.iloc[i - ma_window : i]
                ma_current = spy_history.mean()
                price_current = spy_price.iloc[i]
                
                # Simple Trend Filter: If Price < MA, reduce exposure (defensive)
                # Or check slope of MA. Here we use Price vs MA for simplicity and robustness
                if price_current < ma_current:
                    market_condition = 0.5 # Reduce exposure by 50% in downtrend
            
            # 4. Assign Weights
            # Equal weight among top N assets, adjusted by market condition
            weight = (1.0 / top_n) * market_condition
            
            # Initialize weights for this date to 0
            self.portfolio_weights.loc[current_date, :] = 0.0
            
            # Set weights for top assets
            self.portfolio_weights.loc[current_date, top_assets] = weight
            
            # Remaining weight (1 - market_condition) is effectively in Cash (uninvested)
            # Note: The framework might require weights to sum to 1. 
            # If so, we might need to allocate the rest to a defensive asset like XLU or XLP
            # But for now, let's assume uninvested cash is allowed or handled.
            # If we MUST sum to 1, we can put the rest in the lowest volatility asset (e.g. XLU)
            
            if market_condition < 1.0:
                # Defensive: Put remaining capital into defensive sectors (XLU, XLP)
                defensive_assets = ['XLU', 'XLP']
                # Check if they are in our asset list
                avail_defensive = [a for a in defensive_assets if a in assets]
                if avail_defensive:
                    defensive_weight = (1.0 - market_condition) / len(avail_defensive)
                    self.portfolio_weights.loc[current_date, avail_defensive] += defensive_weight

        # Set weight to 0 for excluded assets (e.g., SPY)
        self.portfolio_weights[self.exclude] = 0
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)

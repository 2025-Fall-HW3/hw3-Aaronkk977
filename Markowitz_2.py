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
    "XLB", # Materials
    "XLC", # Communication Services
    "XLE", # Energy
    "XLF", # Financial
    "XLI", # Industrials
    "XLK", # Technology
    "XLP", # Consumer Staples
    "XLRE",# Real Estate
    "XLU", # Utilities
    "XLV", # Health Care
    "XLY", # Consumer Discretionary
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
    def __init__(self, price, exclude, lookback=126, gamma=0):
        # lookback = 126 (約半年) 用於計算動能
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Initialize weights dataframe
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )
        self.portfolio_weights.fillna(0, inplace=True)

        # Use global Bdf to calculate MA200 with full history
        spy_prices = Bdf['SPY']
        spy_ma200 = spy_prices.rolling(window=200).mean()
        
        # Determine market regime (Bull/Bear) based on t-1 data
        # Bear: SPY < MA200
        # Bull: SPY >= MA200
        # Shift by 1 to use yesterday's data for today's decision
        is_bear_series = (spy_prices < spy_ma200).shift(1)
        
        # Align signal with the portfolio timeframe
        period_signal = is_bear_series.reindex(self.price.index)
        
        current_xlk_weight = 0.0
        last_week = None
        was_bear = False 
        
        # Iterate through dates to apply logic
        for date in self.price.index:
            is_bear = period_signal.loc[date]
            
            # Handle potential NaN
            if pd.isna(is_bear):
                is_bear = False 
            
            current_week = date.isocalendar().week
            
            if not is_bear: # Bull Market
                current_xlk_weight = 1.0
                was_bear = False
            else: # Bear Market
                if not was_bear:
                    # Just switched to Bear -> Clear positions
                    current_xlk_weight = 0.0
                    was_bear = True
                else:
                    # Already in Bear, check for new week to DCA
                    if current_week != last_week:
                        current_xlk_weight += 0.05
                        if current_xlk_weight > 1.0:
                            current_xlk_weight = 1.0
            
            # Assign weight to XLK
            if 'XLK' in self.portfolio_weights.columns:
                self.portfolio_weights.loc[date, 'XLK'] = current_xlk_weight
            
            last_week = current_week
        
        # Ensure excluded asset is 0
        if self.exclude in self.portfolio_weights.columns:
            self.portfolio_weights[self.exclude] = 0

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
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

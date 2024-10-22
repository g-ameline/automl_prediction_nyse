# Project Summary
The goal is to create a financial strategy that outperforms the S&P 500 using machine learning.   
You'll process financial data, build a machine learning model to generate signals,   
and then use these signals to create and backtest a trading strategy.

# Main Steps 

## Data Processing and Feature Engineering

Split data into train (before 2017) and test sets
Create features like 
  - Bollinger Bands, 
  - RSI 
  - MACD
Ensure there's no data leakage in feature creation


## Machine Learning Pipeline

- Implement cross-validation (at least 10 folds, >2 years history each)
- Use either Blocking or Time Series split for cross-validation
- Train models and select the best one based on train set performance
- Save the selected model and its hyperparameters


## Generate Machine Learning Signal

- Use the chosen cross-validation method to generate signals
- Concatenate predictions on validation sets to build the ML signal


## Strategy Backtesting

- Convert ML signal into a financial strategy (e.g., long-only, long-short)
- Compute strategy performance metrics (PnL, max drawdown)
- Create visualizations comparing strategy to S&P 500
- Write a report explaining the approach and results



Key Concepts to Understand:

Time series data handling and the importance of preventing data leakage
Financial indicators (Bollinger Bands, RSI, MACD)
Cross-validation techniques for time series data
Machine learning model selection and evaluation
Converting ML predictions into actionable trading strategies
Backtesting and evaluating trading strategies

Potential Inconsistencies/Questions:

The instructions mention using HistoricalData.csv and all_stocks_5yr.csv, but the repository structure shows only sp500.csv. Clarification on the exact input data might be needed.
The instructions suggest saving certain files (e.g., ml_metrics_train.csv), but these aren't explicitly mentioned in the repository structure. It's unclear if these should be additional files or if they replace some of the listed files.
The audit specifications mention checking if the test set wasn't used for model training and selection, but there's no explicit instruction to hold out the test set in the main instructions. This should be emphasized in the workflow.
The instructions mention optional tasks like training an RNN/LSTM, which should be overlooked as per your request.
The audit specifications seem more detailed in some areas than the original instructions, particularly regarding the specifics of the strategy implementation and backtesting. Students might need more guidance on these aspects in the main instructions.

    Strategy Types:

    The instructions briefly mention examples of strategies (long only, long and short, stock picking) without much detail.
    The audit specifications expect a more specific implementation, including binary, ternary, and proportional strategies. They also mention stock picking, which involves taking long positions on the k best assets and shorting the k worst assets.


    Signal Conversion:

    The instructions don't provide detailed guidance on how to convert the ML signal into a strategy.
    The audit specifications are very specific about this process. They expect the ML signal to be transformed (into long only, long short, binary, ternary, stock picking, or proportional to probability) and then multiplied by the return between d+1 and d+2.


    Investment Amount:

    The instructions don't mention specifics about the investment amount.
    The audit specifications state that you should invest the same amount of money every day (with some exceptions noted for certain strategies).


    PnL Calculation:

    The instructions don't provide details on how to calculate the Profit and Loss (PnL).
    The audit specifications explicitly state that the PnL should be computed as: strategy * future_return.


    Strategy Representation:

    The instructions don't specify how the strategy should be represented.
    The audit specifications expect the strategy to give the amount invested at time t on asset i.


    Plot Details:

    While the instructions mention creating a PnL plot, they don't provide specifics.
    The audit specifications are very detailed about what should be included in the strategy.png plot, including specific axes, scales, and the inclusion of a line separating train and test sets.


    Report Contents:

    The instructions briefly mention creating a markdown report.
    The audit specifications provide a more detailed list of what should be included in the report, such as details on features used, pipeline components, cross-validation specifics, and strategy description.



    These discrepancies suggest that students might need more detailed guidance in the main instructions, particularly regarding:

    Specific strategies to implement and how to convert ML signals into these strategies
    Detailed requirements for the PnL calculation and plot
    Specific expectations for the contents of the final report

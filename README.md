# AutoML Model Prediction on New York Stock Exchange

This project is about training an AI model to predict the next day's price for each ticker (stock company).

## In a nutshell:
  - Download and save data as CSV.
  - Load data as a time series DataFrame.
  - Split data into nontest and test data sets (before and after 2017).
  - Train (quickly) a time series AutoGluon model on 10 different folds of the nontest dataset.
  - Select the best fold (split into training and validation).
  - Perform deeper training on the selected fold with a new model.
  - Define a strategy (picking only the best-predicted ticker).
  - Evaluate the model and strategy with backtesting on the test dataset.
  - Compare the cumulative return against a passive strategy on the S&P.

## Repository structure:

The main work is in the notebook [./notebooks/automl.ipynb] or its PDF [./notebooks/automl.pdf] or its HTML [./notebooks/automl.html].  
There is also [./notebooks/gradient_booster.ipynb] and [./notebooks/neural_network.ipynb], which experiment with gradient boosting and neural networks.

You can also run the notebooks yourself on your machine directly or within a container, but keep in mind that the AutoGluon one is resource-intensive/slow [./how_to_run_it.md].

## Author
gameline


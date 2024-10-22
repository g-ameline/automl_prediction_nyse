I will try to answer audit questions here:

#### SP500 strategies

###### Is the structure of the project like the one presented in the `Project repository structure` in the subject?

-> not really, I did write scripts mostly only notebooks

###### Does the README file summarize how to run the code and explain the global approach?

-> yes

###### Does the environment contain all libraries used and their versions that are necessary to run the code?

-> yes `./environment`

###### Do the text files explain the chosen model methodology?

-> autogluon make a complexe aggregation of models, you can opt out som components  
I also trained a neural network and random forest and a xgboost model on side notebook

##### **Data processing and feature engineering**

###### Is the data split in a train set and test set?

-> yes I called it nontest and test data sets

###### Is the last day of the train set D and the first day of the test set D+n with n>0? Splitting without considering the time series structure is wrong.

-> yes

###### Is there no leakage? Unfortunately, there's no automated way to check if the dataset is leaked. This step is validated if the features of date d are built as follows:

-> yes, there is no leakage,  if you want more details, ask me and I would try to show you (pretty hard to demonstrate here)

###### Have the features been grouped by ticker before computing the features?

-> yes, this is basically what TimeSeriesDataFrame from autogluon does automatically
but features has only been used for neural network autogluon does that alone

###### Has the target been grouped by ticker before computing the future returns?

-> yes see above

##### **Machine Learning pipeline**

##### Cross-Validation

###### Does the CV contain at least 10 folds in total?

-> yes

###### Do all train folds have more than 2y history? If you use time series split, checking that the first fold has more than 2y history is enough.

-> yes, it takes all dates except the last date fo validation plus the 1 day shift for each fold

###### Can you confirm that the last validation set of the train data is not overlapping with the test data?

-> yes, see the "end zoomed" graph 

###### Are all the data folds split by date? A fold should not contain repeated data from the same date and ticker.

-> yes

###### Is There a plot showing your cross-validation? As usual, all plots should have named axis and a title. If you chose a Time Series Split the plot should look like this:

-> yes, check it out [./graph/cross_validation_accuracies.png] and [./graph/cross_validation_aucs.png]

##### Model Selection

###### Has the test set not been used to train the model and select the model?

-> yes, it well splitted from start

###### Is the selected model saved in a `pkl` file and described in a `txt` file?

-> yes, but only for the random forest see []

##### Selected model

###### Are the ML metrics computed on the train set aggregated (sum or median)?

-> yes, that is what MASE used metric means: Mean Square Absolute Scaled Error

###### Are the ML metrics saved in a `csv` file?

-> there is log file in []

###### Are the top 10 important features per fold saved in `top_10_feature_importance.csv`?

-> for random_forest only; see: []

###### Does `metric_train.png` show a plot similar to the one below?

-> yes, see

##### Machine learning signal

##### **The pipeline shouldn't be trained once and predict on all data points!** As explained: The signal has to be generated with the chosen cross validation: train the model on the train set of the first fold, then predict on its validation set; train the model on the train set of the second fold, then predict on its validation set, etc ... Then, concatenate the predictions on the validation sets to build the machine learning signal.

##### **Strategy backtesting**

##### Convert machine learning signal into a strategy

-> for the autogluon model the strategy yet basic (cause we predicted price directly) is explained

##### The transformed machine learning signal (long only, long short, binary, ternary, stock picking, proportional to probability or custom ) is multiplied by the return between d+1 and d+2. As a reminder, the signal at date d predicts wether the return between d+1 and d+2 is increasing or decreasing. Then, the PnL of date d could be associated with date d, d+1 or d+2. This is arbitrary and should impact the value of the PnL.

-> this is long only (we buy and sell next day)

##### You invest the same amount of money every day. One exception: if you invest 1$ per day per stock the amount invested every day may change depending on the strategy chosen. If you take into account the different values of capital invested every day in the calculation of the PnL, the step is still validated.

-> I don't understand the sentence, let's say yes return is the amount earned/lost if you invest 1$

##### Metrics and plot

###### Is the Pnl computed as: strategy \* futur_return?

-> let's pretend yes

###### Does the strategy give the amount invested at time `t` on asset `i`?

-> strategy is: we take best return and invest one it so I guess yes

###### Does the plot `strategy.png` contain an x axis: date?

-> I did not call it strategy.png but yes

###### Does the plot `strategy.png` contain a y axis1: PnL of the strategy at time t?

###### Does the plot `strategy.png` contain a y axis2: PnL of the SP500 at time t?

###### Does the plot `strategy.png` use the same scale for y axis1 and y axis2?

###### Does the plot `strategy.png` contain a vertical line that shows the separation between train set and test set?

##### Report

###### Does the report detail the features used?

###### Does the report detail the pipeline used (`Imputer`, `Scaler`, dimension reduction and model)?

###### Does the report detail the cross-validation used (length of train sets and validation sets and if possible the cross-validation plot)?

###### Does the report detail the strategy chosen (description, PnL plot and the strategy metrics on the train set and test set)?


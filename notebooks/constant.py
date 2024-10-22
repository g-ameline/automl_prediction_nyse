import datetime
import os
import numpy
import matplotlib.pyplot


data_url = 'https://assets.01-edu.org/ai-branch/project4/project04-20221031T173034Z-001.zip'
# data_folder_path = os.path.join('.','data')
root_folder_path='..'
data_folder_name = 'data'
data_folder_path = os.path.join(root_folder_path,data_folder_name)
downloaded_data_file_name = 'downloaded_data.zip'
downloaded_data_file_path= os.path.join(data_folder_path, downloaded_data_file_name)
unzipped_data_folder_name = 'unzipped_data'
unzipped_data_folder_path = os.path.join(data_folder_path, unzipped_data_folder_name)

unzipped_sub_folder_name = 'project04'
unzipped_sub_folder_path = os.path.join(data_folder_path, unzipped_data_folder_name, unzipped_sub_folder_name)
unzipped_stocks_file_name = 'all_stocks_5yr.csv'
unzipped_stocks_file_path = os.path.join(unzipped_data_folder_path, unzipped_sub_folder_name, unzipped_stocks_file_name) 
unzipped_sp500_file_name = 'HistoricalPrices.csv'
unzipped_sp500_file_path = os.path.join(unzipped_data_folder_path, unzipped_sub_folder_name, unzipped_sp500_file_name) 

graph_folder_name = 'graph'
graph_folder_path = os.path.join(root_folder_path,graph_folder_name)
model_folder_name = 'model'
model_folder_path = os.path.join(root_folder_path,model_folder_name)

time_series_split_file_name = 'time_series_split.png'
time_series_split_file_path= os.path.join(graph_folder_path, time_series_split_file_name )
cross_validation_file_name = 'cross_validation.png'
cross_validation_data_file_path= os.path.join(graph_folder_path, cross_validation_file_name )

class ColumnNames:
    date='date'
    price='price'
    ticker='ticker'

needed_stocks_column_names = ['date', 'close', 'Name']
new_stocks_column_names = [ColumnNames.date, ColumnNames.price, ColumnNames.ticker ]
stocks_autogluon_series_column_names = ['item_id', "timestamp"]
stocks_past_covariate_column_names = ['open', 'high', 'low', 'volume']
stocks_data_legacy_column_names_to_autogluon_data_frame_coulumn_names={
    'close':'target',
    'Name': "item_id",
    'date': "timestamp",
    'open':'past_covariate_open',
    'high':'past_covariate_high',
    'low':'past_covariate_low',
    'volume':'past_covariate_volume',
}

sp500_data_legacy_column_names_to_autogluon_data_frame_coulumn_names={
    ' Close':'target',
    'Date': "timestamp",
}

new_stocks_column_names = [ColumnNames.date, ColumnNames.price, ColumnNames.ticker ]

needed_sp500_column_names = ['Date', ' Close' ]
new_sp500_column_names = [ColumnNames.date, ColumnNames.price ]

stocks_file_name = 'stocks.csv'
stocks_file_path = os.path.join(data_folder_path, stocks_file_name)
sp500_file_name = 'sp500.csv'
sp500_file_path = os.path.join(data_folder_path, sp500_file_name)

time_series_stocks_file_name = 'time_series_stocks.csv'
time_series_stocks_file_path= os.path.join(data_folder_path, time_series_stocks_file_name)
time_series_sp500_file_name = 'time_series_sp500.csv'
time_series_sp500_file_path= os.path.join(data_folder_path, time_series_sp500_file_name)
# stocks_static_features_file_name = 'stocks_static_features.csv'
# stocks_static_features_file_path= os.path.join(data_folder_path, stocks_static_features_file_name)

stocks_date_format = '%Y-%m-%d'
iso_date_format = '%Y-%m-%d'
sp500_date_format = "%m/%d/%y" 
test_start_date = '2017-01-01'
test_start_day_time_64 = numpy.datetime64(test_start_date, 'D' )
epoch_date = '2013-01-01'
epoch_day_time_64 = numpy.datetime64(epoch_date, 'D') 
utmost_date = '2020-01-01'
utmost_day_time_64 = numpy.datetime64(utmost_date, 'D') 
test_splitting_date = '2017-01-01'
test_splitting_day_time_64 = numpy.datetime64(test_splitting_date, 'D' )

# class Sp500ColumnNames:
#     date='Date'
#     price=' Close'

max_length_prices = 2000

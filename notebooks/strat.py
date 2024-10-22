import utils
import pandas
import numpy
import os
import constant
import day
import matplotlib.pyplot

def trim_column_names(data_frame):
    return data_frame.rename(
        columns={column: column.strip() for column in data_frame.columns}
    )

def saved_data_frame_path_from_data_frame(data_frame, data_folder_path, data_frame_file_name, save_index=True):
    if not os.path.exists(data_folder_path):
        print(f"\ncreating data_folder: {data_folder_path}")
        os.makedirs(data_folder_path)
    file_path = os.path.join(data_folder_path, data_frame_file_name)
    data_frame.to_csv(file_path, index=save_index)
    return file_path

def saved_data_file_paths_from_url(
    url=constant.data_url,
    redownload=True,
    reextract=True,
):
    utils.show_folders_and_files(constant.data_folder_path)
    if not os.path.exists(constant.data_folder_path):
        print(f"\ncreating data_folder: {constant.data_folder_path}")
        os.makedirs(constant.data_folder_path)

    print(f"\ndownloading raw data from: {constant.data_url}")
    print(f"at: {constant.downloaded_data_file_path}")
    utils.fetch_file_stream(
        url=constant.data_url,
        destination_folder_path=constant.data_folder_path,
        data_file_name=constant.downloaded_data_file_name,
        redownload=redownload,
    )
    print(f"\nunizpping raw data from: {constant.downloaded_data_file_path}")
    print(f"into: {constant.unzipped_data_folder_path}")
    utils.unzip_files(
        zipped_file_path=constant.downloaded_data_file_path,
        destination_folder_path=constant.unzipped_data_folder_path,
        reextract=reextract,
    )
    utils.show_folders_and_files(constant.data_folder_path)

    print(f"\nloading data from: {constant.unzipped_stocks_file_path}")
    print(f"as data frame")
    stocks_data_frame = pandas.read_csv(constant.unzipped_stocks_file_path)
    print(f"keeping only the columns {constant.needed_stocks_column_names}")
    stocks_data_frame = stocks_data_frame[constant.needed_stocks_column_names]
    print(f"rename stocks column from: {constant.needed_stocks_column_names} to {constant.new_stocks_column_names}")
    stocks_data_frame = stocks_data_frame.rename(columns={
        old_name:new_name for old_name, new_name 
        in zip(constant.needed_stocks_column_names, constant.new_stocks_column_names)
    })
    print(f"\n reorder stocks data frame in chronolical order")
    stocks_data_frame = stocks_data_frame.sort_values(by=[constant.ColumnNames.ticker, constant.ColumnNames.date], ascending=True)
    print(f"removing tickers missing latest date price (outdated)")
    stocks_data_frame = past_tickers_removed_data_frame_from_from_data_frame(stocks_data_frame)
    day.day_time_64_unique_sequence_from_data_frame(stocks_data_frame) # will assert valeus are indexed chronologically
    print(f"{stocks_data_frame.head()}")
        
    print(f"\nloading data from: {constant.unzipped_sp500_file_path}")
    print(f"as data frame")
    sp500_data_frame = pandas.read_csv(constant.unzipped_sp500_file_path)
    print(f"keeping only the columns {constant.needed_sp500_column_names}")
    sp500_data_frame = sp500_data_frame[constant.needed_sp500_column_names]
    print(f"rename sp500 column from: {constant.needed_sp500_column_names} to {constant.new_sp500_column_names}")
    sp500_data_frame = sp500_data_frame.rename(columns={
        old_name:new_name for old_name, new_name 
        in zip(constant.needed_sp500_column_names, constant.new_sp500_column_names)
    })
    print(f"{sp500_data_frame.columns = }")
    print(f"\n convert sp500 date values into date time object {sp500_data_frame[constant.ColumnNames.date].iloc[0]}")
    sp500_data_frame[constant.ColumnNames.date] = pandas.to_datetime(
        sp500_data_frame[constant.ColumnNames.date],
        format=constant.sp500_date_format,
    )
    print(f"\n reorder sp500 data frame in chronolical order")
    sp500_data_frame = sp500_data_frame.sort_values(by=[constant.ColumnNames.date], ascending=True)
    sp500_data_frame[constant.ColumnNames.date] = sp500_data_frame[constant.ColumnNames.date].dt.strftime(constant.iso_date_format) 
    print(f"\n converted sp500 date time object into iso string {sp500_data_frame[constant.ColumnNames.date].iloc[0]}")
    print(f"{sp500_data_frame.head()}")
    
    day.day_time_64_unique_sequence_from_data_frame(sp500_data_frame) # will assert valeus are indexed chronologically
    
    print(f"\ndelete zipped and unzipped data")
    utils.delete_everything_inside_folder(folder_path=constant.unzipped_data_folder_path)
    utils.delete_empty_folder(folder_path=constant.unzipped_data_folder_path)
    utils.delete_file(file_path=constant.downloaded_data_file_path)

    print(f"\nsave stocks data at {constant.stocks_file_path}")
    stocks_file_path = saved_data_frame_path_from_data_frame(
        stocks_data_frame,
        constant.data_folder_path,
        constant.stocks_file_name,
    )

    print(f"\nsave sp500 data at {constant.sp500_file_path}")
    sp500_file_path = saved_data_frame_path_from_data_frame(
        sp500_data_frame,
        constant.data_folder_path,
        constant.sp500_file_name,
    )

    print(f"\ndone {stocks_file_path = } {sp500_file_path = }")
    utils.show_folders_and_files(constant.data_folder_path)
    return stocks_file_path, sp500_file_path 


def saved_time_series_data_file_paths_from_url(
    redownload=True,
    reextract=True,
):
    utils.show_folders_and_files(constant.data_folder_path)
    if not os.path.exists(constant.data_folder_path):
        print(f"\ncreating data_folder: {constant.data_folder_path}")
        os.makedirs(constant.data_folder_path)

    print(f"\ndownloading raw data from: {constant.data_url}")
    print(f"at: {constant.downloaded_data_file_path}")
    utils.fetch_file_stream(
        url=constant.data_url,
        destination_folder_path=constant.data_folder_path,
        data_file_name=constant.downloaded_data_file_name,
        redownload=redownload,
    )
    print(f"\nunizpping raw data from: {constant.downloaded_data_file_path}")
    print(f"into: {constant.unzipped_data_folder_path}")
    utils.unzip_files(
        zipped_file_path=constant.downloaded_data_file_path,
        destination_folder_path=constant.unzipped_data_folder_path,
        reextract=reextract,
    )
    utils.show_folders_and_files(constant.data_folder_path)

    print(f"\nloading data from: {constant.unzipped_stocks_file_path}")
    print(f"as data frame")
    stocks_data_frame = pandas.read_csv(constant.unzipped_stocks_file_path)
    print(f"rename stocks column according to: {constant.stocks_data_legacy_column_names_to_autogluon_data_frame_coulumn_names}")
    stocks_data_frame = stocks_data_frame.rename(
        columns=constant.stocks_data_legacy_column_names_to_autogluon_data_frame_coulumn_names,
    )
    print(f"{stocks_data_frame.head()}")
        
    utils.show_folders_and_files(constant.data_folder_path)

    print(f"\nloading data from: {constant.unzipped_sp500_file_path}")
    print(f"as data frame")
    sp500_data_frame = pandas.read_csv(constant.unzipped_sp500_file_path)
    sp500_data_frame['item_id'] = 'sp500'
    print(f"rename sp500 column according to: {constant.sp500_data_legacy_column_names_to_autogluon_data_frame_coulumn_names}")
    sp500_data_frame = sp500_data_frame.rename(
        columns=constant.sp500_data_legacy_column_names_to_autogluon_data_frame_coulumn_names,
    )
    
    print(f"{sp500_data_frame.head()}")
        
    print(f"\ndelete zipped and unzipped data")
    utils.delete_everything_inside_folder(folder_path=constant.unzipped_data_folder_path)
    utils.delete_empty_folder(folder_path=constant.unzipped_data_folder_path)
    utils.delete_file(file_path=constant.downloaded_data_file_path)

    print(f"\nsave stocks data at {constant.stocks_file_path}")
    stocks_file_path = saved_data_frame_path_from_data_frame(
        stocks_data_frame,
        constant.data_folder_path,
        constant.time_series_stocks_file_name,
        save_index=False
    )

    print(f"\nsave sp500 data at {constant.sp500_file_path}")
    sp500_file_path = saved_data_frame_path_from_data_frame(
        sp500_data_frame,
        constant.data_folder_path,
        constant.time_series_sp500_file_name,
        save_index=False
    )

    print(f"\ndone {stocks_file_path = } {sp500_file_path = }")
    utils.show_folders_and_files(constant.data_folder_path)
    
    return stocks_file_path, sp500_file_path  

def data_frame_from_data_file_path(data_file_path):
    data_frame = pandas.read_csv(
        data_file_path,
        index_col=0,
        dtype={
            constant.ColumnNames.price: numpy.float32,
            # constant.ColumnNames.date: numpy.x
        },
    )

    data_frame[constant.ColumnNames.date] = pandas.to_datetime(data_frame[constant.ColumnNames.date]).values.astype('datetime64[D]')
    # opened_day_time_64s = day.get_opened_day_time_64s()
    # assert all(
    #     (day_time_64 in opened_day_time_64s)
    #     for day_time_64  in data_frame[constant.ColumnNames.date].values   
    # ), f"{numpy.sort(data_frame[constant.ColumnNames.date].values)[0] = } {numpy.sort(data_frame[constant.ColumnNames.date].values)[-1] = }"
    # assert constant.epoch_day_time_64 in opened_day_time_64s

    # data_frame[constant.ColumnNames.date] = pandas.to_datetime(data_frame[constant.ColumnNames.date])
    # data_frame[constant.ColumnNames.date] = pandas.Series([date.to_datetime64() for date in data_frame[constant.ColumnNames.date] ])
    # data_frame[constant.ColumnNames.date] = day.day_time_64_series_from_iso_dates_series(data_frame[constant.ColumnNames.date])
    return data_frame

def data_frame_from_data_file_path(data_file_path):
    data_frame = pandas.read_csv(
        data_file_path,
        index_col=0,
        dtype={
            constant.ColumnNames.price: numpy.float32,
            # constant.ColumnNames.date: numpy.x
        },
    )

    data_frame[constant.ColumnNames.date] = pandas.to_datetime(data_frame[constant.ColumnNames.date])
    return data_frame

def ticker_names_from_data_frame(data_frame):
    return numpy.sort(data_frame[constant.ColumnNames.ticker].unique())

def ticker_data_from_data_frame_and_ticker_name(data_frame, ticker_name):
    return data_frame[data_frame[constant.ColumnNames.ticker]== ticker_name]

# let' s say we have a split data set that we want to predict/trin for next day prices;
# we do not need the tickers whose price at day (latest) is not yet/anymore present
def only_ongoing_tickers_from_data_frame(data_frame):
    latest_day_time_64 = data_frame[constant.ColumnNames.date].max()
    ongoing_tickers = data_frame[
        data_frame[constant.ColumnNames.date] == latest_day_time_64 
    ][constant.ColumnNames.ticker]
    return ongoing_tickers

def past_tickers_removed_data_frame_from_from_data_frame(data_frame):
    needed_tickers = only_ongoing_tickers_from_data_frame(data_frame)
    return data_frame[data_frame[constant.ColumnNames.ticker].isin(needed_tickers)]

def splitees_data_frames_from_date(data_frame, before_excluded_splitting_day_time_64=constant.test_splitting_day_time_64):
    assert isinstance(before_excluded_splitting_day_time_64, numpy.datetime64), f"{type(before_excluded_splitting_day_time_64) = }"
    before_date_data_frame = data_frame[data_frame[constant.ColumnNames.date] < before_excluded_splitting_day_time_64]
    since_data_frame = data_frame[data_frame[constant.ColumnNames.date] >= before_excluded_splitting_day_time_64]
    return before_date_data_frame, since_data_frame 

def slice_data_frame_from_first_and_last_dates(data_frame, first_and_last_day_time_64_couple):
    first_day_time_64 = first_and_last_day_time_64_couple[0]
    last_day_time_64 = first_and_last_day_time_64_couple[1]
    slice_data_frame = data_frame[
        (first_day_time_64 <= data_frame[constant.ColumnNames.date]) &
        (data_frame[constant.ColumnNames.date] <= last_day_time_64) 
    ]
    return slice_data_frame


# Function to plot the Time Series Split
def plot_time_series_split(time_series_cross_validator, 
    dates, 
    folder_path=None, 
    file_name=None,
):
    if not os.path.exists(folder_path):
        print(f"\ncreating data_folder: {folder_path}")
        os.makedirs(folder_path)

    fig, ax = matplotlib.pyplot.subplots()
    for i, (train_index, test_index) in enumerate(time_series_cross_validator.split(dates)):
        indices = numpy.arange(len(dates))
        ax.plot(indices[train_index], [i] * len(train_index), 'b.', label='Train' if i == 0 else "")
        ax.plot(indices[test_index], [i] * len(test_index), 'r.', label='Test' if i == 0 else "")
    ax.set_xlabel('Sample index')
    ax.set_ylabel('CV iteration')
    ax.set_title('Time Series Split')
    ax.legend(loc='best')
    matplotlib.pyplot.grid(True)
    if folder_path is not None and file_name is not None:
        if not os.path.exists(folder_path):
            print(f"\ncreating data_folder: {folder_path}")
            os.makedirs(folder_path)
        matplotlib.pyplot.savefig(os.path.join(folder_path,file_name))
    matplotlib.pyplot.show()


def plot_windows(train_windows, 
    forecast_windows, 
    folder_path=None, 
    file_name=None,
):
    figure, axes = matplotlib.pyplot.subplots()
    for index, (train_window, forecast_window) in enumerate(zip(train_windows, forecast_windows)):
        indices = numpy.arange(len(dates))
        axis.plot(
            training_window,
            [index]*len(training_window),
            marker="o",
            c="blue",
            label="Training Window" if window_index == 0 else ""
        )
        axis.plot(indices[train_index], [i] * len(train_index), 'b.', label='Train' if i == 0 else "")
        axis.plot(indices[forecast_index], [i] * len(forecast_index), 'r.', label='forecast' if i == 0 else "")
    axis.set_xlabel('Sample index')
    axis.set_ylabel('CV iteration')
    axis.set_title('Time Series Split')
    axis.legend(loc='best')
    matplotlib.pyplot.grid(True)
    if folder_path is not None and file_name is not None:
        if not os.path.exists(folder_path):
            print(f"\ncreating data_folder: {folder_path}")
            os.makedirs(folder_path)
        matplotlib.pyplot.savefig(os.path.join(folder_path,file_name))
    matplotlib.pyplot.show()


def plot_cv_results(train_accuracies, 
    validate_accuracies, 
    train_auccuracies, 
    validate_auccuracies, 
    folder_path=None, 
    file_name=None,
):
    matplotlib.pyplot.figure(figsize=(14, 7))
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.plot(train_accuracies, marker='o', linestyle='--', label='Train Accuracy')
    matplotlib.pyplot.plot(validate_accuracies, marker='o', linestyle='-', label='Validation Accuracy')
    matplotlib.pyplot.title('Cross-Validation Accuracy')
    matplotlib.pyplot.xlabel('Fold')
    matplotlib.pyplot.ylabel('Accuracy')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)

    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.plot(train_auccuracies, marker='o', linestyle='--', label='Train AUC')
    matplotlib.pyplot.plot(validate_auccuracies, marker='o', linestyle='-', label='Validation AUC')
    matplotlib.pyplot.title('Cross-Validation AUC')
    matplotlib.pyplot.xlabel('Fold')
    matplotlib.pyplot.ylabel('AUC')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)
    
    if folder_path is not None and file_name is not None:
        if not os.path.exists(folder_path):
            print(f"\ncreating data_folder: {folder_path}")
            os.makedirs(folder_path)
        matplotlib.pyplot.savefig(os.path.join(folder_path,file_name))

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(file_name)
    matplotlib.pyplot.show()
   

def plot_multiple_series_separate_axes(
    x, 
    y_series_list, 
    names=None, 
    curve_labels=None, 
    y_labels=None, 
    title='Multiple Series Plot', 
    xlabel='X-axis',
):
    num_plots = len(y_series_list)
    fig, axes = matplotlib.pyplot.subplots(num_plots, 1, figsize=(8, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]
    for i, (ax, y) in enumerate(zip(axes, y_series_list)):
        name = names[i] if names is not None and i < len(names) else f'Series {i+1}'
        curve_label = curve_labels[i] if curve_labels is not None and i < len(curve_labels) else name
        y_label = y_labels[i] if y_labels is not None and i < len(y_labels) else 'Y-axis'
        ax.plot(x, y, label=curve_label)
        ax.set_title(f"{name} Plot")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend()
    matplotlib.pyplot.suptitle(title, fontsize=16)
    matplotlib.pyplot.tight_layout(rect=[0, 0, 1, 0.95])
    matplotlib.pyplot.show()

def plot_multiple_series(
    x,
    y_series_list,
    labels=None,
    title='Multiple Series Plot',
    xlabel='X-axis',
    ylabel='Y-axis',
    folder_path=None,
    file_name=None,
 ):
    matplotlib.pyplot.figure(figsize=(10, 6))
    
    # Plot each y series
    for i, y in enumerate(y_series_list):
        label = labels[i] if labels is not None and i < len(labels) else f'Series {i+1}'
        matplotlib.pyplot.plot(x, y, label=label)
    
    # Adding title and labels
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel(ylabel)
    
    # Show legend if labels are provided
    if labels is not None:
        matplotlib.pyplot.legend()
    
    # Display the plot
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.show() 

    if folder_path is not None and file_name is not None:
        if not os.path.exists(folder_path):
            print(f"\ncreating data_folder: {folder_path}")
            os.makedirs(folder_path)
        matplotlib.pyplot.savefig(os.path.join(folder_path,file_name))

def plot_windows(train_windows, forecast_windows , folder_path, file_name):
    if not os.path.exists(folder_path):
        print(f"\ncreating data_folder: {folder_path}")
        os.makedirs(folder_path)

    figure, axes = matplotlib.pyplot.subplots()
    for window_index, (train_window, forecast_window) in enumerate(zip(train_windows, forecast_windows)):
        axes.plot(
            train_window,
            ([window_index]*len(train_window)),
            marker="o",
            c="blue",
            label="training Window" if window_index == 0 else ""
        )
        axes.plot(
            forecast_window,
            ([window_index+0.2]*len(forecast_window)),
            marker="o",
            c="red",
            label="forecast Window" if window_index == 0 else ""
        )
    axes.set_xlabel('Sample index')
    axes.set_ylabel('CV iteration')
    axes.set_title('Time Series Split')
    axes.legend(loc='best')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.savefig(os.path.join(folder_path,file_name))
    figure.autofmt_xdate()
    matplotlib.pyplot.show()


def create_grouped_bar_chart(
        group_labels, 
        group1_data,
        group2_data,
        group1_label,
        group2_label,
        y_axis_label,
        chart_title,
        folder_path=None,
        file_name=None,
    ):
    assert len(group_labels) == len(group1_data) == len(group2_data), "All input lists must have the same length"
    assert all(isinstance(x, (int, float)) for x in group1_data + group2_data), "All data values must be numbers"
    x_positions = numpy.arange(len(group_labels))  # the label locations
    bar_width = 0.35  # the width of the bars
    figure, axis = matplotlib.pyplot.subplots()
    bars_group1 = axis.bar(x_positions - bar_width/2, group1_data, bar_width, label=group1_label)
    bars_group2 = axis.bar(x_positions + bar_width/2, group2_data, bar_width, label=group2_label)
    axis.set_ylabel(y_axis_label)
    axis.set_title(chart_title)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(group_labels)
    axis.legend()
    def add_value_labels(bar_container):
        for bar in bar_container:
            bar_height = bar.get_height()
    add_value_labels(bars_group1)
    add_value_labels(bars_group2)
    figure.autofmt_xdate()
    figure.tight_layout()
    axis.set_ylim(
        min(min(group1_data),min(group1_data))*0.8,
        max(max(group1_data),max(group1_data))*1.2, 
     )
    if folder_path is not None and file_name is not None:
        if not os.path.exists(folder_path):
            print(f"\ncreating data_folder: {folder_path}")
            os.makedirs(folder_path)
        matplotlib.pyplot.savefig(os.path.join(folder_path,file_name))
    matplotlib.pyplot.show()
    return figure


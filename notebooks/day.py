import pandas_market_calendars
import datetime
import constant
import numpy
import pandas

# why, how:
    # dataframe have iso format day date
        # we covnert it asap
    # numpy like it datetime64 format
        # manipulation will happen with that
    # we will input delta day integerish floats to the model
        # deltas days where last day of sequence is zero

# date verification
def is_valid_date(date_text, date_format=constant.iso_date_format):
    try:
        datetime.datetime.strptime(date_string, date_format)
        return True
    except:
        return False

# conversion
def day_time_64_unique_sequence_from_data_frame(data_frame):
    all_dates = data_frame[constant.ColumnNames.date].unique()
    
    def day_time_64s_from_iso_dates(iso_dates):
        day_time_64s= numpy.array(
            iso_dates, 
            dtype='datetime64[D]',
        )
        assert numpy.all(day_time_64s[:-1] < day_time_64s[1:]), f"we need to verify that values are already chronological {iso_dates}"
        return day_time_64s
    
    all_day_time_64s = day_time_64s_from_iso_dates(all_dates)
    return all_day_time_64s

# def sp500_data_frame_with_iso_dates_from_sp500_data_frame(data_frame):
#     data_frame[constant.ColumnNames.date] = data_frame[constant.ColumnNames.date].assign(iso_date_from_sp500_date)
#     return data_frame

def day_time_64_from_iso_date(iso_date):
    return numpy.datetime64(iso_date, 'D')
# def iso_date_from_sp500_date(sp500_date):
#     return datetime.datetime.strptime(sp500_date, constant.sp500_date_format).isoformat()
def iso_day_from_day_time_64(day_time_64):
     return numpy.datetime_as_string(day_time_64, unit='D')

# def day_time64s_from_iso_date(date_time_index):
#     return numpy.array(date_time_index.index.to_pydatetime(), dtype='datetime64[D]')

# check validity of day according to nyse schedule
def get_opened_day_time_64s(start_date=constant.epoch_date, end_date=constant.utmost_date):
    nyse = pandas_market_calendars.get_calendar('NYSE')
    opened_day_time_index = nyse.valid_days(start_date=start_date, end_date=end_date)
    opened_day_time_64s = opened_day_time_index.values.astype('datetime64[D]')
    opened_day_time_64s = numpy.insert(opened_day_time_64s, 0, constant.epoch_day_time_64) 
    # opened_day_time_64s = numpy.append(opened_day_time_64s, [constant.epoch_day_time_64]) 
    opened_day_time_64s = numpy.insert(opened_day_time_64s, -1, constant.utmost_day_time_64) 
    # opened_day_time_64s = numpy.append(opened_day_time_64s, [constant.utmost_day_time_64]) 
    opened_day_time_64s = numpy.sort(opened_day_time_64s) 
    # numpy.sort(
    #     numpy.insert(
    #         numpy.array([constant.epoch_day_time_64]),opened_day_time_index.values,[constant.utmost_day_time_64],
    #     )
    # ).astype('datetime64[D]')
    return opened_day_time_64s

def is_opened_day_time_64s(day_time_64, opened_day_time_64s=get_opened_day_time_64s()):
    return day_time_64 in opened_day_time_64s

def day_time_64_series_from_iso_dates_series(iso_date_series):
    date_day_time_64_series = pandas.to_datetime(
        iso_date_series,
        format=constant.iso_date_format,
    )
    return date_day_time_64_series 

def next_opened_day_time_64_from_day_time_64(day_time_64, opened_day_time_64s=get_opened_day_time_64s()):
    def value_index_from_value_and_value_sequence(value, value_sequence):
        value_coordinates = (value_sequence == value).nonzero()
        assert len(value_coordinates) == 1, f"{value_coordinates = }"
        value_index = value_coordinates[0][0]
        return value_index

    day_time_64 = day_time_64.astype('datetime64[D]')    
    assert len(opened_day_time_64s) == len(set(opened_day_time_64s)),\
        f"must be an ordered set {len(opened_day_time_64s) = } {len(set(opened_day_time_64s)) = }"
    assert day_time_64 in opened_day_time_64s,\
        f"{day_time_64 = }: {type(day_time_64)} not in {opened_day_time_64s = }"
    next_opened_day_index = value_index_from_value_and_value_sequence(day_time_64, opened_day_time_64s) + 1
    next_opened_day_time_64 = opened_day_time_64s[next_opened_day_index]
    return next_opened_day_time_64

def two_next_opened_trading_day_time_64(from_day_time_64, opened_day_time_64s=get_opened_day_time_64s()):
    next_day_time_64 = next_opened_day_time_64_from_day_time_64(from_day_time_64, opened_day_time_64s)
    next_next_day_time_64 = next_opened_day_time_64_from_day_time_64(next_day_time_64, opened_day_time_64s)
    return next_day_time_64, next_next_day_time_64

# is any missing opened day ?
def is_missing_opened_day_from_day_time_64_sequence(day_time_64_sequence):
    return all(
        next_opened_day_time64_from_day_time_64(day_time_64_sequence[i]) == day_time_64_sequence[i+1]
        for i in range(len(day_time_64_sequence)-2)
    )

if __name__ == '__main__':
    def test():
        pass
    test()

# delta days

delta_day_type = numpy.float32

def delta_day_from_delta_time_64(delta_time_64):
    # assert type(delta_time_64) == numpy.datetime64 ,f"{type(delta_time_64) = }"
    return delta_day_type(delta_time_64 / numpy.timedelta64(1, 'D'))

def delta_days_from_delta_time_64s(delta_time_64s):
    # assert numpy.issubdtype(delta_time_64s.dtype, numpy.datetime64),f"{delta_time_64s.dtype = }"
    return numpy.fromiter(
        (
            delta_day_from_delta_time_64(delta_time_64)
            for delta_time_64 in delta_time_64s
        ),
        dtype = delta_day_type,
    )

def aposteriori_delta_days_from_date_time_64s(date_time_64s, substractive_index=-1):
    # assert numpy.issubdtype(delta_time_64s.dtype, numpy.datetime64),f"{delta_time_64s.dtype = }"
    date_time_64s = numpy.sort(date_time_64s)
    first_non_nat_date = date_time_64s[date_time_64s != numpy.datetime64("NaT")][0]
    for index, date_time_64 in enumerate(date_time_64s):
        if numpy.isnat(date_time_64):
            date_time_64s[index] = first_non_nat_date
    # date_time_64s[date_time_64s == numpy.datetime64("NaT")] = first_non_nat_date
    last_date_time_64 = date_time_64s[substractive_index]
    for date_time_64 in date_time_64s:
        assert not numpy.isnat(date_time_64), f"{date_time_64s = }"
    delta_days = delta_days_from_delta_time_64s(date_time_64s - last_date_time_64)
    for delta_day in delta_days:
        assert not numpy.isnan(delta_day), f"{delta_days = }"
    # print(f"{date_time_64s= }")
    # print(f"{delta_days = }")
    return delta_days
    
# def delta_days_from_date_time_64s_and_all_date_time_64s()

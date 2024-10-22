import constant
import numpy

price_type = numpy.float64

def prices_from_data_frame(data_frame):
    prices = data_frame[constant.ColumnNames.price]
    return prices.values

def aposteriori_relative_prices_from_prices(absolute_prices):
    last_value = absolute_prices[-1]
    if numpy.isnan(last_value):
        return numpy.zeros_like(absolute_prices)
    assert last_value > 0, f"{absolute_prices[-20:] = } {absolute_prices = }"
    scaled_prices = absolute_prices / last_value
    assert scaled_prices[-1] == 1, f"last value of the scaled prices: {scaled_prices[-1]} is not one {scaled_prices = }"
    scaled_prices = numpy.nan_to_num(scaled_prices, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled_prices


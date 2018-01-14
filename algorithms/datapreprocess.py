import pandas as pd
from sklearn import preprocessing


def normalize_data(house_data_frame):
    x = house_data_frame.values
    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(x)
    return pd.DataFrame(x_scaled)


def adjust_for_weights(normalized_df):
    normalized_df[0] = 1 - normalized_df[0]
    normalized_df[3] = 1 - normalized_df[3]
    normalized_df[4] = 1 - normalized_df[4]
    return normalized_df

from DataProcessing import DataReading
from DataProcessing import DataPreprocess
from datamanupulator import Calculations
import numpy as np

df = DataReading.read_data("../Data/HouseDetails.csv")
normalized_df = DataPreprocess.normalize_data(df)

processed_df = DataPreprocess.adjust_for_weights(normalized_df)

house_matrix = Calculations.getMatrix(normalized_df)
weights = np.array([8,6,5,2,3])
weighted_attributes = Calculations.getWeightedAttributeMatrix(house_matrix , weights)

sum = weighted_attributes.sum(axis=1)

get_top_decisions = Calculations.getindexesdecending(sum, 10)
print(get_top_decisions)

get_top_crieterias = Calculations.getindexesdecending(weights,5)
print(get_top_crieterias)

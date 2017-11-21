from DataProcessing import DataReading
from DataProcessing import DataPreprocess
from datamanupulator import Calculations
import numpy as np

# reading data
df = DataReading.read_data("../Data/HouseDetails.csv")
# Pre-processing data
normalized_df = DataPreprocess.normalize_data(df)
processed_df = DataPreprocess.adjust_for_weights(normalized_df)

# normalized house matrix
house_matrix = Calculations.getMatrix(processed_df)
number_of_decisions = len(house_matrix)

# For testing purposes
weights = np.array([8,6,5,4,1])
weighted_attributes = Calculations.getWeightedAttributeMatrix(house_matrix , weights)
sum = weighted_attributes.sum(axis=1)
# Get top 10 decisions
get_top_decisions = Calculations.getindexesdecending(sum, 10)

# Brute force algorithm implementation

# prepare weights in descending order
get_top_criterias = Calculations.getindexesdecending(weights,5)

# Copy of house matrix for constructing new filtering matrix
constructed_house_matrix = house_matrix

for i in range(len(weights)):
    # Get current feature column
    current_feature = constructed_house_matrix[:,get_top_criterias[i]]
    # sort it and get indexes in descending order
    # TODO fix this magical number 18. it is the total-needed rows. rest rows are deleted
    selected_decisions = np.squeeze(np.asarray(current_feature)).argsort()[::-1][number_of_decisions-(i+1)*18:]
    constructed_house_matrix = np.delete(constructed_house_matrix, (selected_decisions), axis=0)

print(constructed_house_matrix.shape)

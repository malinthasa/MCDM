from DataProcessing import DataReading
from DataProcessing import DataPreprocess
from datamanupulator import Calculations
import numpy as np
import os.path
import matplotlib.pyplot as plt

# reading data
df = DataReading.read_data(os.path.join(os.path.split(__file__)[0], "../resources/data/HouseDetails.csv"))
# Pre-processing data
normalized_df = DataPreprocess.normalize_data(df)
processed_df = DataPreprocess.adjust_for_weights(normalized_df)
processed_df.insert(5, 'ID', range(0, len(df)))

# normalized house matrix
house_matrix = Calculations.getMatrix(processed_df)
number_of_decisions = len(house_matrix)

results = np.zeros((1000,))

for iterator in range(0,1000):
    # For testing purposes
    weights = np.random.uniform(low=1, high=10, size=(5,))
    # TODO 0 is appended to the tail of the weight array to store original indexes
    weights = np.insert(weights,len(weights),0)
    weighted_attributes = Calculations.getWeightedAttributeMatrix(house_matrix , weights)
    sum = weighted_attributes.sum(axis=1)
    # Get top 10 decisions
    get_top_decisions = Calculations.getindexesdecending(sum, 10)

    # prepare weights in descending order
    get_top_criterias = Calculations.getindexesdecending(weights,6)

    # Copy of house matrix for constructing new filtering matrix
    constructed_house_matrix = house_matrix

    for i in range(len(weights)-1):
        # Get current feature column
        current_feature = constructed_house_matrix[:,get_top_criterias[i]]
        # sort it and get indexes in descending order
        # TODO fix this magical number 18. it is the total-needed rows. rest rows are deleted
        selected_decisions = np.squeeze(np.asarray(current_feature)).argsort()[::-1][number_of_decisions-(i+1)*18:]
        constructed_house_matrix = np.delete(constructed_house_matrix, (selected_decisions), axis=0)

    # constructed_house_matrix[:,5] contains top 10 decisions found in brute force algorithm
    # get_top_decisions contains top 10 decisions found from standard weight based method

    number_of_correct_decisions = np.sum(np.in1d(constructed_house_matrix[:,5], get_top_decisions)==True)
    performance = number_of_correct_decisions / 10 *100
    results[iterator] = performance

plt.plot(results)
plt.show()

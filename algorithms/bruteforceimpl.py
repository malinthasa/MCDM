from dataprocessing import datareading
from dataprocessing import datapreprocess
from datamanupulator import calculations
import numpy as np
import os.path
import matplotlib.pyplot as plt

# reading data
df = datareading.read_data(os.path.join(os.path.split(__file__)[0], "../resources/data/HouseDetails.csv"))
# Pre-processing data
normalized_df = datapreprocess.normalize_data(df)
processed_df = datapreprocess.adjust_for_weights(normalized_df)
processed_df.insert(5, 'ID', range(0, len(df)))

# normalized house matrix
house_matrix = calculations.get_matrix(processed_df)
number_of_decisions = len(house_matrix)

results = np.zeros((500,))

filter_levels= [90,75,55,35,10]

for iterator in range(0,500):
    # Generating random weights
    weights = np.random.uniform(low=1, high=10, size=(5,))
    # fix weights set
    # weights = np.array([8,5,4,3,1])
    # TODO 0 is appended to the tail of the weight array to store original indexes
    weights = np.insert(weights,len(weights),0)
    weighted_attributes = calculations.get_weighted_attribute_matrix(house_matrix, weights)
    sum = weighted_attributes.sum(axis=1)
    # Get top 10 decisions
    get_top_decisions = calculations.get_indexes_descending(sum, 10)

    # prepare weights in descending order
    get_top_criterias = calculations.get_indexes_descending(weights, 6)

    # Copy of house matrix for constructing new filtering matrix
    constructed_house_matrix = house_matrix

    for i in range(len(weights)-1):
        # Get current feature column
        current_feature = constructed_house_matrix[:,get_top_criterias[i]]
        # sort it and get indexes in descending order
        # TODO fix this magical number 18. it is the total-needed rows. rest rows are deleted
        # here we remove 18 from the tail in each iteration
        selected_decisions = np.squeeze(np.asarray(current_feature)).argsort()[::-1][number_of_decisions-(i+1)*18:]
        # manually picking filtering levels
        # selected_decisions = np.squeeze(np.asarray(current_feature)).argsort()[::-1][filter_levels[i]:]
        constructed_house_matrix = np.delete(constructed_house_matrix, (selected_decisions), axis=0)

    # constructed_house_matrix[:,5] contains top 10 decisions found in brute force algorithm
    # get_top_decisions contains top 10 decisions found from standard weight based method

    number_of_correct_decisions = np.sum(np.in1d(constructed_house_matrix[:,5], get_top_decisions)==True)
    performance = number_of_correct_decisions / 10 *100
    results[iterator] = performance

plt.plot(results)
plt.show()

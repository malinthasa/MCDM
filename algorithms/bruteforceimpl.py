from dataprocessing import datareading
from dataprocessing import datapreprocess
from datamanupulator import calculations
import numpy as np
import os.path
import matplotlib.pyplot as plt

filtering_combinations = np.asarray(list(calculations.generate_all_combinations(5,90)))
reporting = filtering_combinations

# Configurations
number_of_iterations = len(filtering_combinations)
print("Number of combinations: ", number_of_iterations)
# Selecting random set of weights
weights = np.array([8,5,4,3,1])
# TODO 0 is appended to the tail of the weight array to store original indexes
weights = np.insert(weights,len(weights),0)
# Generating random weights
# weights = np.random.uniform(low=1, high=10, size=(5,))
results = np.zeros((number_of_iterations,))
# prepare weights in descending order
get_top_criterias = calculations.get_indexes_descending(weights, 6)

# reading data
df = datareading.read_data(os.path.join(os.path.split(__file__)[0], "../resources/data/HouseDetails.csv"))

# Pre-processing data
normalized_df = datapreprocess.normalize_data(df)
processed_df = datapreprocess.adjust_for_weights(normalized_df)
processed_df.insert(5, 'id', range(0, len(df)))

# normalized house matrix
house_matrix = calculations.get_matrix(processed_df)
number_of_decisions = len(house_matrix)

# Select Top 10 decisions using Weighted Model Method
weighted_attributes = calculations.get_weighted_attribute_matrix(house_matrix, weights)
sum = weighted_attributes.sum(axis=1)
# Get top 10 decisions
get_top_decisions = calculations.get_indexes_descending(sum, 10)

# Select Top 10 decisions using brute force method
for iterator in range(0,number_of_iterations):
    print("Current iteration: ", iterator)
    # Copy of house matrix for constructing new filtering matrix
    constructed_house_matrix = house_matrix

    # Generating filtering criteria for 5 attributes
    # We prepare a list of numbers foe selecting values from each column ramdomly
    # total of the selection is equal to 90 values
    # filter_levels = calculations.generate_filtering_criteria(5, 90)
    # Iterating through each attribute
    for i in range(len(weights)-1):
        # Get current feature column
        current_feature = constructed_house_matrix[:,get_top_criterias[i]]
        # sort it and get indexes in descending order
        if filtering_combinations[iterator][i] !=0:
            selected_decisions = np.squeeze(np.asarray(current_feature)).argsort()[::-1][-filtering_combinations[iterator][i]:]
            constructed_house_matrix = np.delete(constructed_house_matrix, selected_decisions, axis=0)

    # constructed_house_matrix[:,5] contains top 10 decisions found in brute force algorithm
    # get_top_decisions contains top 10 decisions found from standard weight based method

    # Performance calculation
    number_of_correct_decisions = np.sum(np.in1d(constructed_house_matrix[:,5], get_top_decisions)==True)
    performance = number_of_correct_decisions * 10
    results[iterator] = performance
    np.hstack([reporting[iterator],performance])

# Visualization
plt.plot(results)
plt.xlabel('Iteration')
plt.ylabel('Performance %')
plt.show()
np.savetxt("results.csv", reporting, delimiter=",")
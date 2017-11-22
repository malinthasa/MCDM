import random
import numpy

def get_matrix(house_df):
    return house_df.as_matrix(columns=None)


def get_weighted_attribute_matrix(house_matrix, weights):
    return house_matrix * weights

def get_indexes_descending(object, numberofindexes):
    return object.argsort()[::-1][:numberofindexes]

def generate_filtering_criteria(n, total):
    dividers = sorted(random.sample(range(1, total), n - 1))
    return numpy.asarray([a - b for a, b in zip(dividers + [total], [0] + dividers)])
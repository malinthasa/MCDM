

def get_matrix(house_df):
    return house_df.as_matrix(columns=None)


def get_weighted_attribute_matrix(house_matrix, weights):
    return house_matrix * weights

def get_indexes_descending(object, numberofindexes):
    return object.argsort()[::-1][:numberofindexes]


def getMatrix(house_df):
    return house_df.as_matrix(columns=None)


def getWeightedAttributeMatrix(house_matrix, weights):
    return house_matrix * weights
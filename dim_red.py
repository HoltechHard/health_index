import pandas as pd
import numpy as np
from pca_pairs import pca_pair_dim_reduction

# read the dataset
dataset = pd.read_excel("dataset/data_points_indexes.xlsx")
print(dataset.head(3))

# pair of variables
x = np.array(dataset.iloc[:, [3, 4]])
print(x)

# perform PCA
pca_pair_dim_reduction(x)

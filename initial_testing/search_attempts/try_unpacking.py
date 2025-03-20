import os
import pickle
import gzip
import numpy as np

data_path = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting"

size = 100
clustering_method = "qmc"
path = os.path.join(data_path, f"cluster_info_{size}_{clustering_method}.pkl")

with gzip.open(path, "rb") as file:
    data = pickle.load(file)



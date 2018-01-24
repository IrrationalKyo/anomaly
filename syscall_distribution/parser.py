import numpy as np
import csv


# dataset is a list of list
def read_data(file_name):
    dataset = []
    with open(file_name,"r") as f:
        reader = csv.reader(f)
        for row in reader:
            row.remove('')
            dataset.append(row)
    return dataset

# print(read_data("SCFD_ALL_Normal.csv"))


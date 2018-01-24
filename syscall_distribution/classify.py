import parser as pr
import pickle
import cluster as cl
import os.path
import numpy as np
import math

def mahalanobis_dist(cov_obj, x):
    data_point = np.zeros((1, len(x)))
    data_point[0] = x
    return math.sqrt(cov_obj.mahalanobis(data_point)[0])


def mahalanobis_vect(cov_obj_list, x):
    result = []
    for i in range(len(cov_obj_list)):
        result.append(mahalanobis_dist(cov_obj_list[i], x))
    return result

def is_nominal(mah_list, threshold):
    for val in mah_list:
        if val <= threshold:
            return True

    return False


def load_and_classify():
    # distance limit
    bound = 1000
    training_sample_size =2420
    test1_sample_size = 1003
    test2_sample_size = -1

    model_name = "./scfd_k20_2420_2.pmodel"

    if not os.path.isfile(model_name):
        data_set = pr.read_data("SCFD_ALL_Normal.csv")[:training_sample_size]
        result = cl.global_kmeans(data_set, 20)
        pickle.dump(result, open(model_name,"wb"))


data_set = pr.read_data("SCFD_ALL_Normal.csv")
data_point = np.asarray(data_set[0], dtype=np.int32)


with open("./scfd_k20_2420.pmodel","rb") as model_file:
    model = pickle.load(model_file)
    with open("./scfd_k20_2420_2.pmodel","rb") as model_file_2:
        model_2 = pickle.load(model_file_2)
        print(np.array_equal(model["covariance"][0].covariance_, model_2["covariance"][0].covariance_))
        print("model: " + str(mahalanobis_dist(model["covariance"][0],data_point)))
        print("model: " + str(mahalanobis_dist(model_2["covariance"][0],data_point)))




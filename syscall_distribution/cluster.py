from sklearn import cluster, datasets
import numpy as np
import sklearn.metrics.pairwise as pwmetric
import math
import matplotlib.pyplot as plt
import parser
import csv
import json
import pickle
from sklearn.covariance import EmpiricalCovariance

# def clustering_error(dataset, centroids, membership):
#
#     total_error = 0
#
#     for data_index in len(dataset):
#         label = membership[data_index]
#         for center_index in len(centroids):
#             if center_index != membership[data_index]:
#                 continue
#             total_error += pwmetric.paired_euclidean_distances(dataset[data_index],centroids[center_index])
#             break
#
#     return total_error


'''
    k: number of clusters
    
    returns result
    
    result={}
'''
def global_kmeans(dataset, k):

    feature_count = len(dataset[0])

    centroids = np.zeros((k, feature_count))
    result = None
    for i in range(k):

        # candidate: a tulple (error value, new centroids, labels)
        candidate = {"val":math.inf, "centroids":None, "labels":None, "model":None}

        print_once = False


        for j in range(len(dataset)):
            centroids[i] = dataset[j]

            # reshape centroids
            if print_once:
                print("once: " + str(centroids[:i+1]))
                print_once = False

            kmeans = cluster.KMeans(i+1, init=centroids[:i+1], n_init=1)
            kmeans.fit(dataset)

            membership = kmeans.labels_
            new_centers = kmeans.cluster_centers_
            clustering_error = kmeans.inertia_

            if candidate["val"] > clustering_error:
                candidate["val"] = clustering_error
                candidate["centroids"] = new_centers
                candidate["labels"] = membership
                candidate["dataset"] = dataset

        # print(candidate["centroids"])
        centroids[:i+1] = candidate["centroids"]
        result = candidate

    result["labels"]=result["labels"].tolist()
    # calculate the covariance
    cov_mat = [[] for x in range(k)]
    # result["covariance"] = emperical_covariance(dataset,)

    for i in range(len(result["labels"])):
        label = result["labels"][i]
        cov_mat[label].append(dataset[i])

    for i in range(k):
        cov_mat[i] = np.asarray(cov_mat[i], dtype=np.float32)
        cov_mat[i] = EmpiricalCovariance().fit(cov_mat[i])

    result["covariance"] = cov_mat

    return result



if __name__ == "__main__":
    # blobs = datasets.make_blobs(n_samples=1000, random_state=8)
    # dataset = blobs[0]
    dataset = parser.read_data("SCFD_ALL_Normal.csv")
    result = global_kmeans(dataset, 20)
    pickle.dump(result, open("scfd_k20.pmodel","wb"))


    '''
    with open("centroids.json","w") as f:
        json.dump({"centroids": result["centroids"]}, f)
        # wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        # wr.writerow(result["centroids"])
    '''

    '''
    cluster = [[] for x in range(8)]

    for i in range(len(dataset)):
        cluster_number= result["labels"][i]
        cluster[cluster_number].append(dataset[i-1])

    plt.scatter(,cluster[0][:,1]) plt.show()
    '''

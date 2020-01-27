import numpy as np
import math
from scipy.spatial import distance
m=3
ratio=0.1
def fuzzy_c_means(clusters,data_set):
    # clusters=5
    data=np.loadtxt("data\\learn"+str(data_set)+".txt")
    data=np.transpose(data)
    data= data[:2]
    data=np.transpose(data)
    size=len(data)
    memberships=np.zeros([size, clusters])
    distances = np.zeros([size, clusters])
    for i in range(size):
        memberships[i][i%clusters]=1
    cluster_centers=np.zeros([clusters, 2])
    # print(memberships)
    memberships_ex_max=1
    diff=1
    iterations=15
    for i in range(iterations):
        # calculation of cluster centers
        for i in range(clusters):
            for j in range(2):
                upperSum = 0
                lowerSum = 0
                for k in range(size):
                    upperSum += (data[k][j]) * (memberships[k][i]) * (memberships[k][i])
                    lowerSum += (memberships[k][i]) * (memberships[k][i])
                cluster_centers[i][j] = upperSum / lowerSum
                # print(f'{cluster_centers[i][j]} is {i} {j}')

        # calculation of distance of each data point from each cluster center
        for i in range(size):
            for j in range(clusters):
                distances[i][j] = math.sqrt(
                    (cluster_centers[j][0] - data[i][0]) ** 2 + (cluster_centers[j][1] - data[i][1]) ** 2)
        # print(distances)

        # update membership value
        # max_i,max_j=np.unravel_index(memberships.argmax(), memberships.shape)
        # memberships_ex_val=memberships[np.argmax(memberships)]
        for i in range(clusters):
            for k in range(size):
                distSum = 0
                for j in range(clusters):
                    distSum += ((distances[k][i]) / (distances[k][j])) ** 2
                memberships[k][i] = 1 / distSum
        # diff=memberships_ex_val-memberships[max_i][max_j]
    return [memberships,cluster_centers]





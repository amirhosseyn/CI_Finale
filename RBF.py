from FCM import fuzzy_c_means
import numpy as np
import math,ploter

def rfb(data_set, num_of_clusters, cluster_radius):
    m=3
    data_set_classes=[0,2,4,5]
    input_data = np.loadtxt("data\\learn" + str(data_set) + ".txt")
    size=len(input_data)
    data = np.transpose(input_data)
    data = data[:2]
    data = np.transpose(data)
    ans=fuzzy_c_means(num_of_clusters,data_set)
    memberships=ans[0]
    # print(memberships)
    cluster_centers=ans[1]

    # print(cluster_centers)
    G=np.zeros([size,num_of_clusters])
    c=np.zeros([num_of_clusters,2,2])
    # print(c)
    # Computing Covariance matrix
    for i in range(num_of_clusters):
        upper_sum=np.zeros([2,2])
        lower_sum=0
        for k in range(size):
            temp=np.zeros([2,2])
            d=data[k]-cluster_centers[i]
            for i1 in range(2):
                for i2 in range(2):
                    temp[i1][i2]=d[i1] * d[i2]
            # print(np.transpose(data[k]-cluster_centers[i]))
            # print(data[k]-cluster_centers[i])
            # print(temp)
            upper_sum+=(memberships[k][i] ** m) * temp
            lower_sum+=memberships[k][i] ** m
        c[i]=upper_sum/lower_sum
    # print(c)
    # Computing G
    for i in range(num_of_clusters):
        for k in range(size):
            temp=np.matmul(np.transpose(data[k]-cluster_centers[i]),np.linalg.inv(c[i]))
            temp=np.matmul(temp,data[k]-cluster_centers[i])
            G[k][i]=math.exp(- cluster_radius * temp)
    # print(G)
    # Building Y
    Y=np.zeros([size,data_set_classes[data_set]])
    for i in range(size):
        Y[i][int(input_data[i][2])-1]=1
    # print(Y)
    # Computing W
    G_transpose=np.transpose(G)
    W=np.matmul(G_transpose,G)
    W=np.linalg.inv(W)
    W=np.matmul(W,G_transpose)
    Weights=np.matmul(W,Y)
    # print(Weights)
    # Computing argmax
    y_ha=np.matmul(G,Weights)
    y_ha=np.argmax(y_ha,axis=1)
    # print(y_ha)
    y_hat=np.zeros([size,data_set_classes[data_set]])
    for i in range(size):
        y_hat[i][y_ha[i]]=1
    # print(y_hat)
    # Accuracy
    ac=Y-y_hat
    ac=np.sign(ac)
    ac=np.abs(ac)
    ac=ac.sum()
    accuracy=1-ac/size
    print(f"M={num_of_clusters} and accuracy is:{accuracy}")
    # Testing!
    test_data= np.loadtxt("data\\test" + str(data_set) + ".txt")
    t_size=len(test_data)
    # print(len(test_data))
    t_data = np.transpose(test_data)
    t_data = t_data[:2]
    t_data = np.transpose(t_data)
    g_prime=np.zeros([size,num_of_clusters])
    # compute g_prime
    for i in range(num_of_clusters):
        for k in range(t_size):
            temp=np.matmul(np.transpose(t_data[k]-cluster_centers[i]),np.linalg.inv(c[i]))
            temp=np.matmul(temp,t_data[k]-cluster_centers[i])
            g_prime[k][i]=math.exp(- cluster_radius * temp)
    # compute y_test
    y_test=np.zeros([t_size,data_set_classes[data_set]])
    for i in range(t_size):
        y_test[i][int(test_data[i][2])-1]=1

    # Computing argmax test
    y_t_ha=np.matmul(g_prime,Weights)
    y_t_ha=np.argmax(y_t_ha,axis=1)
    # print(y_ha)
    y_hat_test=np.zeros([t_size,data_set_classes[data_set]])
    for i in range(t_size):
        y_hat_test[i][y_t_ha[i]]=1
    # Accuracy of TEST
    ac=y_test-y_hat_test
    ac=np.sign(ac)
    ac=np.abs(ac)
    ac=ac.sum()
    accuracy=1-ac/size
    print(f"M={num_of_clusters} and accuracy of test is:{accuracy}")

    for i in range(t_size):
        if(y_hat_test[i][int(test_data[i][2])-1]!=1):
            test_data[i][2]=-2
    ploter.plot(input_data)
    ploter.plot_c(cluster_centers)
    # ploter.save("fig\\learn_data_set_"+str(data_set)+"_with_m_"+str(num_of_clusters)+"and_radius_"+str(cluster_radius))
    # ploter.show()
    ploter.plot(input_data)
    ploter.plot(test_data)
    ploter.plot_c(cluster_centers)
    ploter.save("fig\\test_data_set_"+str(data_set)+"_with_m_"+str(num_of_clusters)+"and_radius_"+str(cluster_radius))
    ploter.show()

    # border

    x1=np.arange(-5,16,0.4)
    x2=np.arange(-3,15,0.4)
    psize=len(x1) * len(x2)
    array=np.zeros([psize,3])
    for i in range(len(x1)):
        # print(i)
        for j in range(len(x2)):
            array[i*len(x2) + j][0]=x1[i]
            array[i*len(x2) + j][1]=x2[j]

    distances = np.zeros([psize, num_of_clusters])
    membership= np.zeros([psize,num_of_clusters])
    for i in range(psize):
        for j in range(num_of_clusters):
            distances[i][j] = math.sqrt(
                (cluster_centers[j][0] - array[i][0]) ** 2 + (cluster_centers[j][1] - array[i][1]) ** 2)

    for i in range(num_of_clusters):
        for k in range(psize):
            distSum = 0
            for j in range(num_of_clusters):
                distSum += ((distances[k][i]) / (distances[k][j])) ** 2
            membership[k][i] = 1 / distSum
    membership=np.argmax(membership,axis=1)
    for i in range (psize):
        array[i][2] = membership[i]
    # print(membership)
    # ploter.plot_clus(array)
    # ploter.save("fig\\border_data_set_"+str(data_set)+"_with_m_"+str(num_of_clusters)+"and_radius_"+str(cluster_radius))
    # ploter.show()

rfb(3,9,0.1)

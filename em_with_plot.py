from common_functions import *
import matplotlib.pyplot as plt
from expect_maximum_clusters import *
import random
import math
import copy
import numpy as np

# this func calcs the prob distribution
# -takes a point and calcs the prob of being in each cluster returns a mat with prob vals of each cluster
def cal_prob(point,clusters):
    # print("E step")
    # print("Calculating prob for point",point)
    p_point_cluster = []
    dim = len(point)
    # print("dimension",dim)
    for x in clusters:
        cluster_mean = x[0]
        cluster_varience = x[1]
        # print("cluster mean and var",cluster_mean,cluster_varience)
        # cluster_prob_a = 1/pow((2*math.pi), -dim/2) * pow(abs(np.linalg.det(cluster_varience)), -1/2)
        # cluster_prob_b = np.exp(-1/2 * (np.dot((np.subtract(point,cluster_mean)).T, np.linalg.inv(cluster_varience)), np.subtract(point,cluster_mean)))
        diff_mat = np.array(np.subtract(point, cluster_mean))[np.newaxis]
        # print("diff mat shape",diff_mat.shape)
        second_mat = np.array(np.linalg.inv(cluster_varience))
        # print("second mat shape", second_mat.shape)
        diff_mat_trans = np.transpose(diff_mat)
        # print("third mat shape", diff_mat_trans.shape)
        probability_1 = 1 / (pow((2 * math.pi), dim *0.5) * pow(abs(np.linalg.det(cluster_varience)),0.5))
        probability_2 = math.exp(float(-0.5 * diff_mat.dot(second_mat).dot(diff_mat_trans)))
        probability = probability_1*probability_2
        # print("prob",probability)
        # print("distance",distance(point,(cluster_mean_x,cluster_mean_y)))
        # print("cluster prob a", cluster_prob_a)
        # print("cluster prob b", cluster_prob_b)
        # cluster_prob = cluster_prob_a*cluster_prob_b
        #print("cluster prob for point",cluster_prob)
        p_point_cluster.append(probability)
    #return p_point_cluster
    b_k_sum = 0
    b_k =[]
    for i in range(0,len(clusters)):
        #print(clusters[i][2])
        b_k_sum = b_k_sum + p_point_cluster[i]*clusters[i][2]
    if(b_k_sum!=0):
        for i in range(0,len(clusters)):
            b_k.append((p_point_cluster[i]*clusters[i][2])/float(b_k_sum))
    else:
        for i in range(0,len(clusters)):
            b_k.append(1/len(clusters))

    # print("likelyhood",b_k)
    # print("sum of likelyhood",sum(b_k))
    if(sum(b_k)<0.99999 or sum(b_k)>1.1):
        print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrroooooooooooooooooooooooooooooooooooooooooorrrrrrrr",sum(b_k))
    return [b_k,np.log(b_k_sum)]

    #print("prob without nomalization",p_point_cluster)
    # sum_cluster = sum(p_point_cluster)
    # p_cluster =[]
    # for x in p_point_cluster:
    #     ans = x/sum_cluster
    #     p_cluster.append(ans)
    # # print("probablities",p_cluster)
    # # print("sum of prob",sum(p_cluster))
    # return p_cluster

# updates the mean,var and weights and sends it as a new cluster
def update_cluster(clusters,prob_dist,points):
    # print("M step")
    prob = np.array(prob_dist)
    copy_points = np.array(copy.deepcopy(points))
    new_clusters = copy.deepcopy(clusters)
    for i in range(0, len(clusters)):
        # print("cluster", i)
        sum_prob_i = 0
        for x in prob:
            # print("prob of point",x)
            sum_prob_i = x[i] + sum_prob_i

        # print("sum of prob for this cluster",sum_prob_i)
        weight_mat = []
        for p in prob:
            weight_mat.append(p[i] / sum_prob_i)
        #print("Weight mat", weight_mat)
        t = 0
        mean_i = np.zeros(len(copy_points[0]))

        for p in copy_points:
            # print("point",p,"prob for cluster",prob[t])
            # print("test",np.array(p)*float(prob[t][i]))
            mean_i = (np.array(p) * float(weight_mat[t])) + mean_i
            # print("mean num",mean_i_num)
            t = t + 1

        # print("mean for clus",i,mean_i)
        var_num = np.zeros([len(copy_points[0]), len(copy_points[0])])
        # print("var num",var_num)
        t = 0
        for p in copy_points:
            diff_mat = p - mean_i[np.newaxis]
            diff_mat_tras = np.transpose(diff_mat)
            # print("diff mat ", diff_mat)
            # print("diff mat shape",diff_mat.shape)
            # print("diff mat trans shape", np.transpose(diff_mat[np.newaxis]).shape)
            var_num = var_num + (float(weight_mat[t]) * diff_mat_tras * diff_mat)
            t = t + 1

        new_clusters[i][0] = mean_i
        new_clusters[i][1] = var_num
        new_clusters[i][2] = sum_prob_i / len(copy_points)

    # print(" old cluster", i, "vals ", clusters)
    # print(" new cluster", i, "updated vals ", new_clusters)
    # print("updated cluster",clusters)
    return new_clusters



def em_clustering(given_points,no_k,bic_bool):

    plt.figure(1)
    ax = plt.subplot()
    # for i in given_points:
    #     ax.plot(i[0],i[1],color='red', marker='o', markersize=2)
    number_of_clusters = no_k
    no_of_dim = len(given_points[0])
    no_points = len(given_points)
    em_clusters = []#this mat will hold[mean_mat,covar_mat,weight] for each cluster
    final_prob_dist =[]#this holds the prob of ech point belonging to each cluster [[p1,p2...pk],[p1,p2,....pk]........all points ]
    best_ll = 0
    clusters = []
    for k in range(number_of_clusters):
        clusters.append([])

    co_var_mat = np.identity(no_of_dim)#initial covar matrix
    if(bic_bool==True):
        ll_threshold = 2
    else:
        ll_threshold = 1
    #co_var_mat = [[1,0],[0,1]]
    #print("co var mat test",co_var_mat)
    weight_cluster = 1/number_of_clusters
    for i in range(0, number_of_clusters):
        random_mean = given_points[random.randint(0, no_points - 1)]
        em_clusters.append([random_mean, co_var_mat, weight_cluster])
        #em_clusters.append([means_k[i],co_var_mat,weight_cluster])
    # print("Initial cluster mean , covar , weight")
    # for e in em_clusters:
    #     print(e)
    # print("running EM")
    run  = True
    iteration = 0
    #number_iterations = 100
    # ax.axis('equal')
    # for e in em_clusters:
    #     #circle1 = plt.Circle((e[0][0], e[0][1]),radius=e[1],color='blue',fill=False)
    #     #ax.add_artist(circle1)
    #     ax.plot(e[0][0], e[0][1], color='blue', marker='s', markersize=7)
    #
    #     #plt.scatter(e[0][0],e[0][1],c = 'b',marker='d',data=[e[0][0],e[0][1]])
    ####### E step ##########################
    while(run):
        prob_dist = []
        temp_ll = 0
        # print("initial em cluster",em_clusters)
        for p in given_points:
            temp = cal_prob(p, em_clusters)
            prob_dist.append(temp[0])
            temp_ll = temp_ll + temp[1]
        # for p in prob_dist:
        #     print(p)
        ####### M step ##########################

        #print("Log Likelyhood iteration",iteration,temp_ll)
        em_clusters = update_cluster(em_clusters, prob_dist, given_points)
        final_prob_dist = prob_dist
        diff_t = abs(temp_ll - best_ll)
        #print("diff", diff_t)
        if (diff_t < ll_threshold and diff_t != float('-inf') and diff_t!=float('inf')):
            run = False
        best_ll = temp_ll
        iteration = iteration+1
        # print("updated clusters mean")
        # for k in em_clusters:
        #     print(k[0])
    point = 0
    for p in final_prob_dist:
        clus = p.index(max(p))
        (clusters[clus]).append(given_points[point])
        point = point+1


    #     #     c = plt.Circle((e[0][0],e[0][1]),e[1],facecolor='None',fill='None',edgecolor='black')
    #
    # print("final cluster",em_clusters)
    # print("clusters")
    # for l in clusters:
    #     print(l)
    # print("final prob dist")
    # for k in final_prob_dist:
    #     print(k)


    return ([best_ll,clusters,em_clusters])




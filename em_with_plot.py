from common_functions import *
import matplotlib.pyplot as plt
from expect_maximum_clusters import *
import random
import math
import copy
import numpy as np

def cal_prob(point,clusters):# this func cal the prob distribution -takes a point and cal the prob of beingh in each cluster returns a mat with prob vals of each cluster
    #print("E step")
    #print("Calculating prob for point",point)
    p_point_cluster = []
    dim = len(point)
    #print("dimension",dim)
    for x in clusters:
        cluster_mean = x[0]
        cluster_varience = x[1]
        #print("cluster mean and var",cluster_mean,cluster_varience)
        # cluster_prob_a = 1/pow((2*math.pi), -dim/2) * pow(abs(np.linalg.det(cluster_varience)), -1/2)
        # cluster_prob_b = np.exp(-1/2 * (np.dot((np.subtract(point,cluster_mean)).T, np.linalg.inv(cluster_varience)), np.subtract(point,cluster_mean)))
        diff_mat = np.array(np.subtract(point, cluster_mean))[np.newaxis]
        #print("diff mat shape",diff_mat.shape)
        second_mat = np.array(np.linalg.inv(cluster_varience))
        #print("second mat shape", second_mat.shape)
        diff_mat_trans = np.transpose(diff_mat)
        #print("third mat shape", diff_mat_trans.shape)
        probability_1 = 1 / (pow((2 * math.pi), dim *0.5) * pow(abs(np.linalg.det(cluster_varience)), 0.5))
        probability_2 = np.exp(float(-0.5 * diff_mat.dot(second_mat).dot(diff_mat_trans)))
        probability = probability_1*probability_2
        #print("prob",probability)
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
    # if(sum(b_k)<0.9999 or sum(b_k)>1):
    #     print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrroooooooooooooooooooooooooooooooooooooooooorrrrrrrr",sum(b_k))
    return b_k

    #print("prob without nomalization",p_point_cluster)
    # sum_cluster = sum(p_point_cluster)
    # p_cluster =[]
    # for x in p_point_cluster:
    #     ans = x/sum_cluster
    #     p_cluster.append(ans)
    # # print("probablities",p_cluster)
    # # print("sum of prob",sum(p_cluster))
    # return p_cluster

def update_cluster(clusters,prob_dist,points):#updates the mean,var and weights and sends it as a new cluster
    #print("M step")
    prob = np.array(prob_dist)
    copy_points = np.array(copy.deepcopy(points))
    new_clusters = copy.deepcopy(clusters)
    for i in range(0,len(clusters)):
        #print("cluster", i)
        sum_prob_i = 0
        for x in prob:
            #print("prob of point",x)
            sum_prob_i = x[i]+sum_prob_i

        #print("sum of prob for this cluster",sum_prob_i)
        t=0
        mean_i_num = np.zeros(len(copy_points[0]))

        for p in copy_points:
            #print("point",p,"prob for cluster",prob[t])
            #print("test",np.array(p)*float(prob[t][i]))
            mean_i_num = (np.array(p)*float(prob[t][i])) + mean_i_num
            #print("mean num",mean_i_num)
            t=t+1
        mean_i = np.array(mean_i_num)/sum_prob_i
        #print("mean for clus",i,mean_i)
        var_num = np.zeros([len(copy_points[0]),len(copy_points[0])])
        #print("var num",var_num)
        t=0
        for p in copy_points:
            diff_mat = p-mean_i[np.newaxis]
            diff_mat_tras = np.transpose(diff_mat)
            # print("diff mat ", diff_mat)
            # print("diff mat shape",diff_mat.shape)
            # print("diff mat trans shape", np.transpose(diff_mat[np.newaxis]).shape)
            var_num = var_num + (float(prob[t][i]) * diff_mat_tras*diff_mat)
            t=t+1
        #print("var num", var_num)
        var = var_num/sum_prob_i
        #print("var of cluster",i, var)
        # var = math.sqrt(var)

        new_clusters[i][0] = mean_i
        new_clusters[i][1] = var
        new_clusters[i][2] = sum_prob_i/len(copy_points)

    # print(" old cluster", i, "vals ", clusters)
    # print(" new cluster", i, "updated vals ", new_clusters)
    #print("updated cluster",clusters)
    return new_clusters


def em_clustering(data_path,no_k):
    given_points = np.array(read_board(data_path))
    plt.figure(2)
    ax = plt.subplot()
    # for i in given_points:
    #     ax.plot(i[0],i[1],color='red', marker='o', markersize=2)
    number_of_clusters = no_k
    no_of_dim = len(given_points[0])
    print("no of clusters",number_of_clusters)
    print("Number of Dimensions",no_of_dim)
    no_points = len(given_points)
    print("NUmber of points",no_points)
    em_clusters = []#this mat will hold[mean_mat,covar_mat,weight] for each cluster
    final_prob_dist =[]#this holds the prob of ech point belonging to each cluster [[p1,p2...pk],[p1,p2,....pk]........all points ]
    co_var_mat = np.identity(no_of_dim)#initial covar matrix
    #co_var_mat = [[1,0],[0,1]]
    #print("co var mat test",co_var_mat)
    weight_cluster = 1/number_of_clusters
    for i in range(0, number_of_clusters):
        random_mean = given_points[random.randint(0, no_points - 1)]
        em_clusters.append([random_mean,co_var_mat,weight_cluster])
    for e in em_clusters:
        print(e)
    number_iterations = 25
    # ax.axis('equal')
    # for e in em_clusters:
    #     #circle1 = plt.Circle((e[0][0], e[0][1]),radius=e[1],color='blue',fill=False)
    #     #ax.add_artist(circle1)
    #     ax.plot(e[0][0], e[0][1], color='blue', marker='s', markersize=7)
    #
    #     #plt.scatter(e[0][0],e[0][1],c = 'b',marker='d',data=[e[0][0],e[0][1]])
    ####### E step ##########################
    for i in range(0, number_iterations):
        prob_dist = []
        # print("initial em cluster",em_clusters)
        for p in given_points:
            prob_dist.append(cal_prob(p, em_clusters))
        # for p in prob_dist:
        #     print(p)
        ####### M step ##########################
        em_clusters = update_cluster(em_clusters, prob_dist, given_points)
        final_prob_dist = prob_dist
        # print("updated clusters mean")
        # for k in em_clusters:
        #     print(k[0])


    #     #     c = plt.Circle((e[0][0],e[0][1]),e[1],facecolor='None',fill='None',edgecolor='black')
    #
    print("final cluster",em_clusters)
    # print("final prob dist")
    # for k in final_prob_dist:
    #     print(k)
    if(no_of_dim==2):
        color_bar = []
        for e in em_clusters:
            f = (random.uniform(0.1, 1), random.uniform(0, 1), random.uniform(0, 1))
            color_bar.append(f)
        l = 0
        for p in final_prob_dist:
            # print(p)
            c = p.index(max(p))
            # print(c)
            ax.plot(given_points[l][0], given_points[l][1], color=color_bar[c], marker='o', markersize=10, alpha=0.4,
                    mec=color_bar[c], mew=0)
            l = l + 1
            # if(c==0):
            #     ax.plot(given_points[l][0], given_points[l][1], color='red', marker='o', markersize=2)
            #     l=l+1
            # elif (c == 1):
            #     ax.plot(given_points[l][0], given_points[l][1], color='cyan', marker='o', markersize=2)
            #     l=l+1
            # else:
            #     ax.plot(given_points[l][0], given_points[l][1], color='magenta', marker='o', markersize=2)
            #     l=l+1
        i = 0
        for e in em_clusters:
            #     # circle2 = plt.Circle((e[0][0], e[0][1]),radius=e[1],color='green',fill=False)
            #     # ax.add_artist(circle2)
            ax.plot(e[0][0], e[0][1], color=color_bar[i], marker='d', markersize=8, markeredgecolor='k',
                    markeredgewidth=1)
            i = i + 1
        #     # print("final circle plotted")
        ax.set_title('EM')






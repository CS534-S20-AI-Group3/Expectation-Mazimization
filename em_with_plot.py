from __future__ import division
from common_functions import *
import matplotlib.pyplot as plt
from expect_maximum_clusters import *
import random
import math
import copy

def cal_prob(point,clusters):
    #print("point passed",point[0]," ",point[1])
    p_point_cluster = []
    for x in clusters:
        cluster_mean_x = x[0][0]
        cluster_mean_y = x[0][1]
        cluster_varience = x[1]
        #print("cluster mean and var",cluster_mean_x,cluster_mean_y,cluster_varience)
        cluster_prob_a = 1/(math.sqrt(2*math.pi*cluster_varience*cluster_varience))
        cluster_prob_b = math.exp(-0.5*(distance(point,(cluster_mean_x,cluster_mean_y))**2)/cluster_varience**2)
        # print("distance",distance(point,(cluster_mean_x,cluster_mean_y)))
        # print("cluster prob a", cluster_prob_a)
        # print("cluster prob b", cluster_prob_b)
        cluster_prob = cluster_prob_a*cluster_prob_b
        #print("cluster prob",cluster_prob)
        p_point_cluster.append(cluster_prob)
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

def update_cluster(clusters,prob,points):
    copy_points = copy.deepcopy(points)
    for i in range(0,len(clusters)):
        #print("cluster", i)
        sum_prob_i = 0
        for x in prob:
            #print("prob of point",x)
            sum_prob_i = x[i]+sum_prob_i
        mean_i_x_num = 0
        mean_i_y_num = 0
        #print("sum of prob for this cluster",sum_prob_i)
        t=0
        for p in copy_points:
            #print("point",p,"prob for cluster",prob[t])
            mean_i_x_num = p[0]*prob[t][i] + mean_i_x_num
            mean_i_y_num = p[1]*prob[t][i] + mean_i_y_num
            #print("mean x num",mean_i_x_num)
            t=t+1
        mean_i_x = mean_i_x_num/sum_prob_i
        mean_i_y = mean_i_y_num/sum_prob_i
        var_num = 0
        t=0
        for p in copy_points:
            var_num = var_num + (prob[t][i]*((distance(p,(mean_i_x,mean_i_y)))**2))
            t=t+1
        var = var_num/sum_prob_i
        var = math.sqrt(var)
        clusters[i][0][0] = mean_i_x
        clusters[i][0][1] = mean_i_y
        clusters[i][1] = var
        clusters[i][2] = sum_prob_i/len(copy_points)
        #print("cluster",i,"updated vals",mean_i_x,mean_i_y,var)
    #print("updated cluster",clusters)
    return clusters


def em_clustering(data_path,no_k):
    given_points = read_board(data_path)
    plt.figure(2)
    ax = plt.subplot()
    # for i in given_points:
    #     ax.plot(i[0],i[1],color='red', marker='o', markersize=2)
    number_of_clusters = no_k
    number_iterations = 500
    no_points = len(given_points)
    print(no_points)
    em_clusters = []
    for i in range(0, number_of_clusters):
        random_mean = given_points[random.randint(0, no_points - 1)]
        em_clusters.append([[random_mean[0], random_mean[1]], 10, 1 / number_of_clusters])
    # ax.axis('equal')
    # for e in em_clusters:
    #     #circle1 = plt.Circle((e[0][0], e[0][1]),radius=e[1],color='blue',fill=False)
    #     #ax.add_artist(circle1)
    #     ax.plot(e[0][0], e[0][1], color='blue', marker='s', markersize=7)
    #
    #     #plt.scatter(e[0][0],e[0][1],c = 'b',marker='d',data=[e[0][0],e[0][1]])
    for i in range(0, number_iterations):
        prob_dist = []
        # print("initial em cluster",em_clusters)
        for p in given_points:
            prob_dist.append(cal_prob(p, em_clusters))
        em_clusters = update_cluster(em_clusters, prob_dist, given_points)
        # print("updated clusters",em_clusters)

        #     c = plt.Circle((e[0][0],e[0][1]),e[1],facecolor='None',fill='None',edgecolor='black')

    color_bar = []
    for e in em_clusters:
        f = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        color_bar.append(f)
    l = 0
    for p in prob_dist:
        # print(p)
        c = p.index(max(p))
        ax.plot(given_points[l][0], given_points[l][1], color=color_bar[c], marker='o', markersize=2)
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
        # circle2 = plt.Circle((e[0][0], e[0][1]),radius=e[1],color='green',fill=False)
        # ax.add_artist(circle2)
        ax.plot(e[0][0], e[0][1], color=color_bar[i], marker='d', markersize=8, markeredgecolor='k', markeredgewidth=1)
        ax.set_title('EM')
        i = i + 1
        # print("final circle plotted")
    for e in em_clusters:
        print(e)
    plt.show()




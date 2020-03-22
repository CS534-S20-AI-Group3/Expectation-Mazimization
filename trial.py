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
        b_k_sum = b_k_sum + p_point_cluster[i]*clusters[i][2]
    for i in range(0,len(clusters)):
        b_k.append((p_point_cluster[i]*clusters[i][2])/b_k_sum)
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








given_points = read_board(r"C:\Users\Ankit\Desktop\sample_EM_data.csv")
for i in given_points:
    plt.scatter(i[0],i[1],c ='r',s =0.5,marker='o')
number_of_clusters = 3
no_points = len(given_points)
print(no_points)
em_clusters =[]
for i in range(0,number_of_clusters):
    random_mean = given_points[random.randint(0,no_points)]
    em_clusters.append([[random_mean[0],random_mean[1]],5,1/number_of_clusters])
for e in em_clusters:
    plt.scatter(e[0][0],e[0][1],c = 'b',marker='d',data=[e[0][0],e[0][1]])
for i in range(0,100):
    prob_dist = []
    print("initial em cluster",em_clusters)
    for p in given_points:
        prob_dist.append(cal_prob(p,em_clusters))
    em_clusters=update_cluster(em_clusters,prob_dist,given_points)
    print("updated clusters",em_clusters)

    #     c = plt.Circle((e[0][0],e[0][1]),e[1],facecolor='None',fill='None',edgecolor='black')
for e in em_clusters:
        plt.scatter(e[0][0],e[0][1],c = 'g')
plt.show()


#!/usr/bin/env python
# coding: utf-8

# In[113]:


import networkx as nx
import matplotlib.pyplot as plt
from string import ascii_lowercase
import pandas as pd
import math
import random
import sys
import re
import itertools 
import xml.etree.ElementTree as ET
from sys import argv
import argparse
from time import sleep
import os
import subprocess
import json
import random
import copy
import time


# In[114]:


name_arg= sys.argv[1]
no_subSet_Arg1= sys.argv[2]
g = nx.read_graphml(name_arg)
x=int(no_subSet_Arg1)

g= nx.Graph(g)
pos = nx.kamada_kawai_layout(g)
print(nx.info(g))
edge_labels = nx.get_edge_attributes(g, 'latency')

# ### Select the starting node with minimum avg of outgoing edges value

# In[117]:



def get_firstnode():
    nodes=list(g.nodes)
    lst_nodes_avg_wgt = pd.DataFrame()

    for node in nodes:
        neighbors_of_Node = g[node]
        #print('node :', node, 'neighbor: ', neighbors_of_Node)
        sum = 0
        count = 0
        for n_node in neighbors_of_Node:
            #print('node n neighbor: ',node,n_node)
            #print( '')
            sum += g.get_edge_data(node,n_node,default=0)['latency']
            count+=1
            
        lst_nodes_avg_wgt = lst_nodes_avg_wgt.append({"node": node, "latency": sum/count}, ignore_index=True)
        lst = lst_nodes_avg_wgt[lst_nodes_avg_wgt.latency == lst_nodes_avg_wgt.latency.min()]
        #print(sum)
    lst.reset_index(drop=True)
    min_wgt_node = lst.iloc[0]
    
    
    #Print all the list
    #print( lst_nodes_avg_wgt)
    #print("Selected Starting Node id: ", min_wgt_node[1],"\nAvg. Edge Wgt: ",  round(min_wgt_node[0],2))
    return min_wgt_node


# In[118]:

# # Alogrithm for selecting subset of required number of nodes

# In[119]:

start_t=time.time()
def subSet_of_Nodes(Num_Of_Nodes):
    firstnode = get_firstnode().node
    currentnode = firstnode
    nodes=list(g.nodes)
    
    #list to maintain the all the visited nodes
    lst_VisitedNodes = pd.DataFrame()
    lst_VisitedNodes = lst_VisitedNodes.append({"node": firstnode, "visited node": firstnode, "latency": 0, "path":'-'}, ignore_index=True)
    #print( lst_VisitedNodes)
    path = firstnode
    lst_Neighbors = pd.DataFrame()
    
    for n in nodes:
        neighbors_of_CurrentNode = g[currentnode]
        if(lst_VisitedNodes.shape[0] >1):

            neighbors_of_CurrentNode = [ele for ele in neighbors_of_CurrentNode if ele not in list(lst_VisitedNodes['visited node'])] 
            nodes = [i for i in nodes if i not in list(lst_VisitedNodes['visited node'])]
            lst_Neighbors = lst_Neighbors[~lst_Neighbors['neighbor'].isin(list(lst_VisitedNodes['visited node']))] 
         
        
        #Add the neighbors of the current node to the list of all neighbors
        for node in neighbors_of_CurrentNode:
            lst_Neighbors = lst_Neighbors.append({"node": currentnode, "neighbor":node, "latency": 
                                                  g.get_edge_data(currentnode,node,default=0)['latency']},  ignore_index=True)
            
            
        
        # Get the weights of all the neighbors of the current node and save them in the list above
        weights_of_neighbors = pd.DataFrame()
        for index, row in lst_Neighbors.iterrows():
            w =  row['latency']
            weights_of_neighbors = weights_of_neighbors.append({"node": row['neighbor'], "latency": w,
                                                                'path': path},  ignore_index=True)
      
        # Check if the list of the neighbor nodes is not empty
        if(weights_of_neighbors.shape[0] >0):
            lst = weights_of_neighbors[weights_of_neighbors.latency == weights_of_neighbors.latency.min()]
            lst.reset_index(drop=True)
            node_with_min_weight = lst.iloc[0]

            # Update the path with the current node's neighbor
            path = path + ',' + node_with_min_weight.node
            if (lst_VisitedNodes.shape[0] > Num_Of_Nodes):
                return lst_VisitedNodes
            
            else:
                # add the node with minimum weight to the visited node list
                lst_VisitedNodes = lst_VisitedNodes.append({"node": currentnode, "visited node": node_with_min_weight.node,
                                                            "latency": node_with_min_weight.latency, 'path': path}, ignore_index=True)
                currentnode = node_with_min_weight.node
                
        else: 
            currentnode = firstnode
            path = firstnode
            
    #return the list of all visited nodes.
        #print(lst_VisitedNodes.latency.max())
    return lst_VisitedNodes

end_t=time.time()

# ### Get Average Delay of the Selected Nodes:

# In[120]:


def getAvgDelay(path_final):
    avg = []
    sum=0
    for x,y in itertools.combinations(path_final, 2):
        sum = getPathDelay(x, y, Final_graph)
        #print( 'sum: ', round(sum,2))
        avg.append(round(sum,2))
    return avg

def getAvgCost(path_final):
    avg = []
    sum=0
    for x,y in itertools.combinations(path_final, 2):
        sum = getPathCost(x, y, Final_graph)
        avg.append(round(sum,2))
    return avg

def getAvgBW(path_final):
    avg = []
    sum=0
    for x,y in itertools.combinations(path_final, 2):
        sum = getPathBW(x, y, Final_graph)
        avg.append(round(sum,2))
    return avg

def getPathDelay(firstNode, EndNode, p_graph):
    path = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
    sub_graph = g.subgraph(path)
    sum = 0
    for edge in sub_graph.edges(data=True):
        lat= edge[2]['latency']
        #lat= edge[2]['cost']

        #print ("Node id: ", edge , lat )
        sum+=lat
    #print('edge sum:', round(sum,2))
    return (sum)

def getPathCost(firstNode, EndNode, p_graph):
    path = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
    sub_graph = g.subgraph(path)
    sum = 0
    for edge in sub_graph.edges(data=True):
        #lat= edge[2]['latency']
        lat= edge[2]['cost']
        sum+=lat
    return (sum)

def getPathBW(firstNode, EndNode, p_graph):
    path = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
    sub_graph = g.subgraph(path)
    sum = 0
    lst =[]
    for edge in sub_graph.edges(data=True):
        lat= edge[2]['bw']
        lst.append(lat)
        #sum+=lat
    return (min(lst))

def hop(path_final, g):
    hop  = []
    for x,y in itertools.combinations(path_final, 2):
        #print(x,y)
        path = nx.shortest_path(g, str(x), str(y))
        hop1=(len(path)-1)
        hop.append(hop1)
    #print(hop)
    return hop


def sum_list(items):
    sum_num = 0
    for x in items:
        sum_num += x
    return sum_num


#Result of the Subset of nodes
nodes=list(g.nodes)


path_final = []
for i in range(0, x):
    node = nodes[i]
    path_final.append(node)

print('selected subset of nodes: ', path_final)

Final_graph = g

#Delays Calculation
AvgDelay=list(getAvgDelay(path_final))
avg_delay_value = round(sum_list(AvgDelay)/len(AvgDelay),2)
min_delay_value = min(AvgDelay)
max_delay_value = max(AvgDelay)

#BW Calculation
AvgBW=list(getAvgBW(path_final))
avg_bw_value = round(sum_list(AvgBW)/len(AvgBW),2)
min_bw_value = min(AvgBW)
max_bw_value = max(AvgBW)

#Cost Calculation
AvgCost=list(getAvgCost(path_final))
#print('len of AvgDelay list ', len(AvgDelay))
avg_cost_value = round(sum_list(AvgCost)/len(AvgCost),2)
min_cost_value = min(AvgCost)
max_cost_value = max(AvgCost)



# #No. of Hop Calculation
Hop_No = hop(path_final,g)
avg_hop_value = round(sum_list(Hop_No)/len(Hop_No),2)
min_hop_value = min(Hop_No)
max_hop_value = max(Hop_No)

#Results Output
print("Selected Starting Node is: ", (path_final[0]))
run_t = ((end_t - start_t )* 1000)
print("running time", round(run_t , 3), "ms") # round(time.time() * 1000
print ('The Required Subset of Nodes:', x)
print('list of Selected Subset of Nodes: ', path_final)
print(' ')
print('List of Delays', AvgDelay)
print('The min latency value in selected subset:', min_delay_value)
print('The max latency value in selected subset:', max_delay_value)
print('The avg latency value in selected subset:', avg_delay_value)
#print('sum of the delay in selected subset:', sum_list(AvgDelay))
print(' ')
print('List of BW', AvgBW)
print('The min bw value in selected subset:', min_bw_value)
print('The max bw value in selected subset:', max_bw_value)
print('The avg bw value in selected subset:', avg_bw_value)
print(' ')
print('List of Cost', AvgCost)
print('The min cost value in selected subset:', min_cost_value)
print('The max cost value in selected subset:', max_cost_value)
print('The avg cost value in selected subset:', avg_cost_value)
print('sum of the cost in selected subset:', round(sum_list(AvgCost),2))
print(' ')
print('List of hops', Hop_No)
print('The min no. of hops traversed in selected subset:', min_hop_value)
print('The max no. of hops traversed in selected subset:', max_hop_value)
print('The avg no. of hops traversed in selected subset:', avg_hop_value)
print(' ')


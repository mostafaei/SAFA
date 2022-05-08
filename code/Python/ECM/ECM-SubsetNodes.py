#!/usr/bin/env python
# coding: utf-8


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

## Delay Calculation Among Nodes for TopologyZoo Networks
class Graph:
    def __init__(self,graphName, g):
        self.graphName = graphName
        self.g = g
    def read_graph(self):
        Topo=str(self.graphName)
        pos = nx.kamada_kawai_layout(self.g)
        print(nx.info(self.g))

        #Delay Calculation Among Nodes for TopologyZoo Networks
        if (Topo=='largeSNR'or Topo=='mediumSNR' or Topo=='smallSNR' or Topo=='largeSAW'or Topo=='mediumSAW' or Topo=='smallSAW'):
            True
        else:
            xml_tree    = ET.parse(Topo+'.graphml')
            namespace   = "{http://graphml.graphdrawing.org/xmlns}"
            ns          = namespace # just doing shortcutting, namespace is needed often.

            #GET ALL ELEMENTS THAT ARE PARENTS OF ELEMENTS NEEDED LATER ON
            root_element    = xml_tree.getroot()
            graph_element   = root_element.find(ns + 'graph')

            # GET ALL ELEMENT SETS NEEDED LATER ON
            index_values_set    = root_element.findall(ns + 'key')
            node_set            = graph_element.findall(ns + 'node')
            edge_set            = graph_element.findall(ns + 'edge')

            # SET SOME VARIABLES TO SAVE FOUND DATA FIRST
            # memomorize the values' ids to search for in current topology
            node_label_name_in_graphml = ''
            node_latitude_name_in_graphml = ''
            node_longitude_name_in_graphml = ''
            # for saving the current values
            node_index_value     = ''
            node_name_value      = ''
            node_longitude_value = ''
            node_latitude_value  = ''
            # id:value dictionaries
            id_node_name_dict   = {}     # to hold all 'id: node_name_value' pairs
            id_longitude_dict   = {}     # to hold all 'id: node_longitude_value' pairs
            id_latitude_dict    = {}     # to hold all 'id: node_latitude_value' pairs

            # FIND OUT WHAT KEYS ARE TO BE USED, SINCE THIS DIFFERS IN DIFFERENT GRAPHML TOPOLOGIES
            for i in index_values_set:

                if i.attrib['attr.name'] == 'label' and i.attrib['for'] == 'node':
                    node_label_name_in_graphml = i.attrib['id']
                if i.attrib['attr.name'] == 'Longitude':
                    node_longitude_name_in_graphml = i.attrib['id']
                if i.attrib['attr.name'] == 'Latitude':
                    node_latitude_name_in_graphml = i.attrib['id']

            # NOW PARSE ELEMENT SETS TO GET THE DATA FOR THE TOPO
            # GET NODE_NAME DATA
            # GET LONGITUDE DATK
            # GET LATITUDE DATA
            for n in node_set:

                node_index_value = n.attrib['id']

                #get all data elements residing under all node elements
                data_set = n.findall(ns + 'data')

                #finally get all needed values
                for d in data_set:

                    #node name
                    if d.attrib['key'] == node_label_name_in_graphml:
                        #strip all whitespace from names so they can be used as id's
                        node_name_value = re.sub(r'\s+', '', d.text)
                    #longitude data
                    if d.attrib['key'] == node_longitude_name_in_graphml:
                        node_longitude_value = d.text
                    #latitude data
                    if d.attrib['key'] == node_latitude_name_in_graphml:
                        node_latitude_value = d.text


                    #save id:data couple
                    id_node_name_dict[node_index_value] = node_name_value
                    id_longitude_dict[node_index_value] = node_longitude_value
                    id_latitude_dict[node_index_value]  = node_latitude_value
                    #print(node_name_value , ' longi ', node_longitude_value, ' lati ', node_latitude_value)
                    #print('long: ', id_longitude_dict)


            for e in edge_set:

                # GET IDS FOR EASIER HANDLING
                e[0] = e.attrib['source']
                e[1] = e.attrib['target']
                #print('e0: ', e[0], ' e1: ', e[1])
                latitude_src= math.radians(float(id_latitude_dict[e[0]]))
                latitude_dst= math.radians(float(id_latitude_dict[e[1]]))
                longitude_src= math.radians(float(id_longitude_dict[e[0]]))
                longitude_dst= math.radians(float(id_longitude_dict[e[1]]))

                first_product               = math.sin(latitude_dst) * math.sin(latitude_src)
                second_product_first_part   = math.cos(latitude_dst) * math.cos(latitude_src)
                second_product_second_part  = math.cos(longitude_dst - longitude_src)
                distance = (math.acos(first_product + (second_product_first_part * second_product_second_part))) * 6378.137

                # t (in ms) = ( distance in km * 1000 (for meters) ) / ( speed of light / 1000 (for ms))
                # t         = ( distance       * 1000              ) / ( 1.97 * 10**8   / 1000         )
                latency = ( distance * 1000 ) / ( 197000 )
                self.g[e[0]][e[1]]['latency']=round(latency,2) 
        return self.g


## Select the Starting Node with Max Num of Neighbors
class getFirstNode:

    def __init__(self, g):
        self.g = g
        
    
    def firstNodeDCM(self):
        nodes=list(self.g.nodes)
        lst_nodes_avg_wgt = pd.DataFrame()
        for node in nodes:
            neighbors_of_Node = self.g[node]
            sum = 0
            count = 0
            for n_node in neighbors_of_Node:
                sum += self.g.get_edge_data(node,n_node,default=0)['latency']
                count+=1          
            lst_nodes_avg_wgt = lst_nodes_avg_wgt.append({"node": node, "latency": sum/count}, ignore_index=True)   
            lst = lst_nodes_avg_wgt[lst_nodes_avg_wgt.latency == lst_nodes_avg_wgt.latency.min()]
        min_wgt_node = lst.iloc[0]
        return min_wgt_node
    
    def firstNodeECM(self):
        nodes=list(self.g.nodes)
        lst_node_neighbors = pd.DataFrame()
        for node in nodes:
            neighbors_of_Node = self.g[node]
            count = 0
            sum = 0
            for n_node in neighbors_of_Node:
                sum += self.g.get_edge_data(node,n_node,default=0)['latency']
                count+=1
                avg=sum/count
                rank=count+(count/avg)
            lst_node_neighbors = lst_node_neighbors.append({"node": node, "nodeCount": count, "rank1": rank}, ignore_index=True)
            lst = lst_node_neighbors[lst_node_neighbors.rank1 == lst_node_neighbors.rank1.max()]
        max_wgt_node = lst.iloc[0]
        return max_wgt_node


## JSON File Topology for P4
class jsonFile:
    
    def __init__(self, g, Final_graph, path):
        self.g = g
        self.path= path
        self.Final_graph = Final_graph
        
    def topologyJSON(self):
        final_obj = {}
        json_obj_hosts = {}
        json_obj_switch = {}
        nodes=list(self.g.nodes)
        nodes = list(map(int, nodes))
        json_obj_links_hosts = []
        json_obj_links_switches = []
        lstPortsCounts = []
        switch_count = 1
        index = 0
        count = 1
        e =1
        h =1
        n =1

        for h in nodes:
            h +=1
            host_ip = "10.0.%d.%d/24" % (int(h) ,int(h))
            host_ip2 = "10.0.%d.%d" % (int(h), int(h+1))
            host_mac = '08:00:00:00:%02x:%02x' % (int(h), int(h))
            host_mac2 = ""
            host_mac2 = '08:00:00:00:%02x:00' % (int(h))
            command = ["route add default gw "+str(host_ip2)+" dev eth0", "arp -i eth0 -s "+str(host_ip2)+" "+str(host_mac2)]
            temp = {"ip":host_ip, "mac":host_mac, "commands":command}
            json_obj_hosts["h"+str(h)] = {"ip":host_ip, "mac":host_mac, "commands":command}
            json_obj_switch['s'+str(count)] = { "runtime_json" : "triangle-topo/s"+str(count)+"-runtime.json" }
            count +=1   

        for n in nodes:
            n +=1
            json_obj_links_hosts.insert(index, ["h"+str(n), "s"+str(switch_count)+"-p1"])
            lstPortsCounts.insert(index, 1)
            index += 1
            switch_count+=1

        for e in list(self.g.edges):
            lat = self.g[e[0]][e[1]]['latency']
            lstPortsCounts[int(e[0])] += 1
            lstPortsCounts[int(e[1])] += 1
            port0 =  lstPortsCounts[int(e[0])]
            port1 =  lstPortsCounts[int(e[1])]
            e = list(e)
            e[0] = str (int(e[0])+1)
            e[1] = str (int(e[1])+1)
            json_obj_links_switches.insert(index, ["s"+e[0]+"-p"+str(port0),"s"+e[1]+"-p"+str(port1),str(float(lat))+"ms"])                                     
            index+=1

        json_obj_links = json_obj_links_hosts+json_obj_links_switches
        final_obj["hosts"] = json_obj_hosts
        final_obj["switches"] = json_obj_switch
        final_obj["links"] = json_obj_links

        with open(self.path + 'topology.json', 'w') as outfile:
            json.dump(final_obj, outfile)
        return json_obj_links_switches

    #JSON File Switches for P4
    def findDirectConnect(startNode, endNode, json_obj_links_switches):
        for obj in json_obj_links_switches:
            if(int(obj[0].split("-")[0].replace("s","")) == startNode and int(obj[1].split("-")[0].replace("s","")) == endNode):
                 return (obj[0].split("-")[1])
            elif(int(obj[1].split("-")[0].replace("s","")) == startNode and int(obj[0].split("-")[0].replace("s","")) == endNode):
                return (obj[1].split("-")[1])

    def getPort(firstNode, EndNode, p_graph, json_obj_links_switches):
        path1 = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
        path = list(map(int, path1)) #list to int
        for x in range (len(path)):
            path[x] +=1
        port = jsonFile.findDirectConnect(int(path[0]), int(path[1]), json_obj_links_switches)
        return port

    def switchJSON(self):
        json_obj_links_switches = jsonFile.topologyJSON(self)
        nodes=self.g.nodes
        nodes = list(map(int, nodes))
        restrictedList = list(self.Final_graph.nodes)
        restrictedList = list(map(int, restrictedList))
        for i in range (len(restrictedList)):
            restrictedList[i] +=1
        Final_List = {}
        count = 1
        h=1
        for h in nodes:
            h+=1
            Final_List = {"target": "bmv2",
            "p4info": "build/basic.p4.p4info.txt",
            "bmv2_json": "build/basic.json",
            "table_entries":[{
              "table": "MyIngress.ipv4_lpm",
              "default_action": True,
              "action_name": "MyIngress.drop",
              "action_params": { }
            }]}
            if int(h) in restrictedList:
                for n in restrictedList:
                    if ( int(h) == int (n)):
                        host_ip = "10.0.%d.%d" % (int(n), int(n))
                        host_mac = ""
                        host_mac = '08:00:00:00:%02x:%02x' % (int(n), int(n))
                        port = int (1)
                        temp = {"table": "MyIngress.ipv4_lpm", 
                                "match":{
                                    "hdr.ipv4.dstAddr":[host_ip, 32]
                                },
                                "action_name": "MyIngress.ipv4_forward",
                                "action_params": {
                                    "dstAddr": host_mac,
                                    "port": port
                                }
                                }
                    else: 
                        host_ip = "10.0.%d.%d" % (int(n), int(n))
                        host_mac = ""
                        host_mac = '08:00:00:00:%02x:00' % (int(n))
                        port = jsonFile.getPort(int(h)-1, n-1, self.Final_graph, json_obj_links_switches).replace("p","")
                        port = int (port)
                        temp = {"table": "MyIngress.ipv4_lpm", 
                                "match":{
                                    "hdr.ipv4.dstAddr":[host_ip, 32]
                                },
                                "action_name": "MyIngress.ipv4_forward",
                                "action_params": {
                                    "dstAddr": host_mac,
                                    "port": port
                                }
                                }
                    Final_List["table_entries"].insert(count,temp)
                    count+=1
            with open(self.path + 's' + str(h) + '-runtime.json', 'w') as outfile:
                json.dump(Final_List, outfile)


## Alogrithm for Selecting Subset of Node Cluster
class nodeSubset:
    
    def __init__(self, Num_Of_Nodes, startNode,g):
        self.Num_Of_Nodes = Num_Of_Nodes
        self.startNode = startNode
        self.g = g

    def subSet_of_Nodes(self):
        firstnode = self.startNode
        currentnode = firstnode
        nodes=list(self.g.nodes)

        #list to maintain the all the visited nodes
        lst_VisitedNodes = pd.DataFrame()
        lst_VisitedNodes = lst_VisitedNodes.append({"node": firstnode, "visited node": firstnode, "latency": 0, "path":'-'}, ignore_index=True)
        path = firstnode
        lst_Neighbors = pd.DataFrame()

        for n in nodes:

            neighbors_of_CurrentNode = self.g[currentnode]
            if(lst_VisitedNodes.shape[0] >1):

                neighbors_of_CurrentNode = [ele for ele in neighbors_of_CurrentNode if ele not in list(lst_VisitedNodes['visited node'])] 
                nodes = [i for i in nodes if i not in list(lst_VisitedNodes['visited node'])]
                lst_Neighbors = lst_Neighbors[~lst_Neighbors['neighbor'].isin(list(lst_VisitedNodes['visited node']))] 

            #Add the neighbors of the current node to the list of all neighbors
            for node in neighbors_of_CurrentNode:
                lst_Neighbors = lst_Neighbors.append({"node": currentnode, "neighbor":node, "latency": 
                                                      self.g.get_edge_data(currentnode,node,default=0)['latency']},  ignore_index=True)


            # Get the weights of all the neighbors of the current node and save them in the list above
            weights_of_neighbors = pd.DataFrame()
            for index, row in lst_Neighbors.iterrows():
                w =  row['latency']
                weights_of_neighbors = weights_of_neighbors.append({"node": row['neighbor'], "latency": w,
                                                                    'path': path},  ignore_index=True)
                #print(weights_of_neighbors)
            # Check if the list of the neighbor nodes is not empty
            if(weights_of_neighbors.shape[0] >0):
                lst = weights_of_neighbors[weights_of_neighbors.latency == weights_of_neighbors.latency.min()]
                lst.reset_index(drop=True)
                node_with_min_weight = lst.iloc[0]

                # Update the path with the current node's neighbor
                path = path + ',' + node_with_min_weight.node
                if (lst_VisitedNodes.shape[0] > self.Num_Of_Nodes):
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
        return lst_VisitedNodes

## Get paramter data from the Graph Topology
class getData:
    
    def getAvgDelay(g, path_final, Final_graph):
        avg = []
        sum=0
        for x,y in itertools.combinations(path_final, 2):
            sum = getData.getPathDelay(g, x, y, Final_graph)
            avg.append(round(sum,2))
        return avg

    def getAvgCost(g, path_final, Final_graph):
        avg = []
        sum=0
        for x,y in itertools.combinations(path_final, 2):
            sum = getData.getPathCost(g, x, y, Final_graph)
            avg.append(round(sum,2))
        return avg

    def getAvgBW(g, path_final, Final_graph):
        avg = []
        sum=0
        for x,y in itertools.combinations(path_final, 2):
            sum = getData.getPathBW(g, x, y, Final_graph)
            avg.append(round(sum,2))
        return avg

    def getPathDelay(g, firstNode, EndNode, p_graph):
        path = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
        sub_graph = g.subgraph(path)
        sum = 0
        for edge in sub_graph.edges(data=True):
            lat= edge[2]['latency']
            sum+=lat
        return (sum)

    def getPathCost(g, firstNode, EndNode, p_graph):
        path = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
        sub_graph = g.subgraph(path)
        sum = 0
        for edge in sub_graph.edges(data=True):
            lat= edge[2]['cost']
            sum+=lat
        return (sum)

    def getPathBW(g, firstNode, EndNode, p_graph):
        path = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
        sub_graph = g.subgraph(path)
        sum = 0
        lst =[]
        for edge in sub_graph.edges(data=True):
            lat= edge[2]['bw']
            lst.append(lat)
        return (min(lst))

    def hop(g, path_final):
        hop  = []
        for x,y in itertools.combinations(path_final, 2):
            path = nx.shortest_path(g, str(x), str(y))
            hop1=(len(path)-1)
            hop.append(hop1)
        return hop

    def sum_list(items):
        sum_num = 0
        for x in items:
            sum_num += x
        return sum_num


## Print paramter data of the Graph Topology
class printData:
    
    def __init__(self, g, reqNodes, startNode, final_path, Final_graph):
        self.g = g
        self.reqNodes = reqNodes
        self.startNode = startNode
        self.final_path = final_path
        self.Final_graph = Final_graph
        
        #Delays Calculation
        AvgDelay=list(getData.getAvgDelay(g, final_path, Final_graph))
        avg_delay_value = round(getData.sum_list(AvgDelay)/len(AvgDelay),2)
        min_delay_value = min(AvgDelay)
        max_delay_value = max(AvgDelay)

        #BW Calculation
        AvgBW=list(getData.getAvgBW(g, final_path, Final_graph))
        avg_bw_value = round(getData.sum_list(AvgBW)/len(AvgBW),2)
        min_bw_value = min(AvgBW)
        max_bw_value = max(AvgBW)

        #Cost Calculation
        AvgCost=list(getData.getAvgCost(g, final_path, Final_graph))
        avg_cost_value = round(getData.sum_list(AvgCost)/len(AvgCost),2)
        min_cost_value = min(AvgCost)
        max_cost_value = max(AvgCost)


        #No. of Hop Calculation
        Hop_No = getData.hop(g, final_path)
        avg_hop_value = round(getData.sum_list(Hop_No)/len(Hop_No),2)
        min_hop_value = min(Hop_No)
        max_hop_value = max(Hop_No)

        #Results Output
        print('The Required Subset of Nodes:', self.reqNodes+1)
        print('Selected Starting Node is: ', self.startNode)
        print('list of Selected Subset of Nodes: ', self.final_path)
        print(' ')
        print('List of Delays', AvgDelay)
        print('The min latency value in selected subset:', min_delay_value)
        print('The max latency value in selected subset:', max_delay_value)
        print('The avg latency value in selected subset:', avg_delay_value)
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
        print('sum of the cost in selected subset:', round(getData.sum_list(AvgCost),2))
        print(' ')
        print('List of hops', Hop_No)
        print('The min no. of hops traversed in selected subset:', min_hop_value)
        print('The max no. of hops traversed in selected subset:', max_hop_value)
        print('The avg no. of hops traversed in selected subset:', avg_hop_value)
        print(' ') 


## Get Average Delay of the Selected Nodes:
## Result of the required Subset of nodes (ECM)
def ECM(n,g):
    req_nodeSubset=int(n)
    startNode=getFirstNode(g)
    startNode=startNode.firstNodeECM().node
    print("startNodeECM:", startNode)
    start_t=time.time()
    listNodes=nodeSubset(req_nodeSubset,startNode,g)
    result=listNodes.subSet_of_Nodes()
    end_t=time.time()
    run_t = ((end_t - start_t )* 1000)
    print("running time", round(run_t , 3), "ms") # round(time.time() * 1000
    final_path=list(result.iloc[-1].tail(2).head(1))
    final_path=final_path[0]
    final_path=final_path.split(",")
    Final_graph = g.subgraph(final_path)
    printData(g, req_nodeSubset, startNode, final_path, Final_graph) 
    path_final  = list(Final_graph.node)
    
    #Provide dir. path to save JSON files for P4
    path=str('JsonFiles/')
    topoJSON=jsonFile(g, Final_graph,path)
    topo=topoJSON.topologyJSON()
    switchRule=topoJSON.switchJSON()

    # "result" shows path of visited nodes
    #print (result)
    
def main():
    #Select a graph from TopoloyZoo
    name_arg= sys.argv[1]
    no_subSet_Arg1= sys.argv[2]
    x=int(no_subSet_Arg1)-1
    topo = str(name_arg.replace(".graphml", ""))
    g = nx.read_graphml(name_arg)
    g = nx.Graph(g)
    g = Graph(topo, g)
    g = g.read_graph()  
    ECM(x,g)





#     #Graph Representation of %s topology' % (topo)
#     pos = nx.kamada_kawai_layout(g)
#     edge_labels = nx.get_edge_attributes(g, 'latency')
#     plt.figure(figsize=(12, 10))
#     nx.draw_networkx_nodes(g,  pos, node_size=300, label=1)
#     nx.draw_networkx_edges(g, pos, width=2, edge_color='black')
#     nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=9)
#     nx.draw_networkx_labels(g, pos, font_size=12)
#     plt.title('Graph Representation of %s topology' % (topo) )
#     plt.show()

    
if __name__ == '__main__':
      main()

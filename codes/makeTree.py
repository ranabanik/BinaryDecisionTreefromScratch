"""
rana.banik@vanderbilt.edu
check Utils.py for functions
"""
import numpy as np
from anytree import Node, NodeMixin, AnyNode, RenderTree
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import json
import os
import glob
from Utils import gini_index, info_gain, computeOptimalSplit, checkLeaf, buildTreeNew
from Utils import TreeNode
from Utils import predict

data_path = r'C:\Users\ranab\OneDrive\PycharmProjects\CS6362_assng_1_cancer_dataset\BinaryDecisionTreefromScratch\cancer_datasets_v2'
data_files = glob.glob(os.path.join(data_path,'*.csv'))

data_files

"""Each of the 569 samples used in the dataset consists of a feature vector of length 30. 
The first 10 entries in this feature vector are the mean of the characteristics listed above for each2 image. 
The second 10 are the standard deviation and last 10 are the largest value of each of these characteristics present in each image. 
Each sample is also associated with a label: a label of value 1 indicates the sample was for malignant (cancerous) tissue. 
A label of value 0 indicates the sample was for benign tissue."""

# train_data1 = np.loadtxt(data_files[2], delimiter=',', skiprows=1)
train_data1 = pd.read_csv(data_files[2])
test_data1 = pd.read_csv(data_files[0])
valid1 = pd.read_csv(data_files[4])

# train_label_1 = train_data1.iloc[:,0] #only the labels (398x1)
# train_data_1 = train_data1.iloc[:,1:] #only the features (398x30)

# pd.DataFrame(train_data1)

"""
get Gini index(impurity) of a node
"""
# gini_index(train_data1)

"""
to get splitting information and child nodes of a parent node
"""
# metadata, left, right = computeOptimalSplit(train_data1)

"""
get information gain of a split
"""
# info_gain(train_data1,left,right)


"""
to check a node is leaf(pure) or not
"""
# checkLeaf(train_data1)

"""
To get all leaf nodes and split
"""
# ln = []
# spinf = []
# buildTree(train_data1,ln,spinf)


"""
Using from here
___________________________________________________________________________
"""



    # return nodeList,splitList,nodeCount

"""
some trials on anytree library
"""

# class MyBaseClass(object):
#     # def __init__(self,object):
#         # self.object = None
#     foo = 4 #??
#
# class MyClass(MyBaseClass, NodeMixin):  # Add Node feature
#     def __init__(self, name, length, width, parent=None, children=None):
#         super(MyClass, self).__init__()
#         self.name = name
#         self.length = length
#         self.width = width
#         self.parent = parent
#         if children:
#             self.children = children
#
# #
# my0 = MyClass('my0',0,0,children=None)
# my1 = MyClass('my1', 1, 0, parent=my0)
# my0 = MyClass('my0',0,0, children=[MyClass('my2',0,2)])
# #
#
# # my0.name
# ###
# root = AnyNode(id="root")
# s0 = AnyNode(id="sub0", parent=root)
# s0b = AnyNode(id="sub0B", parent=s0, foo=4, bar=109) #if you do not give parent, s0b.parent = None
# # s0b = AnyNode(id="sub0B", foo=4, bar=109)



# class TreeNode(NodeMixin):
#     def __init__(self,node_name,dataframe, feat_name,feat_val,parent = None, children=None):
#         super(TreeNode,self).__init__()
#         self.node_name = node_name
#         self.dataframe = dataframe
#         self.feat_name = feat_name
#         self.feat_val = feat_val
#         if parent:
#             self.parent = parent
#         if children:
#             self.children = children
#
#     def classify(self,datapoint):
#         if len(self.children)==0:
#             if (np.sum(self.dataframe.iloc[:,0])/len(self.dataframe.iloc[:,0])) >= 0.5:
#                 # print(self.feat_name)
#                 return 1
#             else:
#                 # print(self.feat_name)
#                 return 0
#         else:
#             if datapoint[self.feat_name] <= self.feat_val:
#                 # print(self.feat_name)
#                 print('Biriani')
#                 return self.children[0].classify(datapoint) #todo Rana
#             else:
#                 # print(self.feat_name)
#                 print('chicken')
#                 return self.children[1].classify(datapoint)



"""
creating root node:
"""

root = TreeNode("root",train_data1,None,None,None,None)

"""
tree new
"""
totNode = 0
nList = []
spList = []
buildTreeNew(root,nList,spList,totNode)

# test = test_data1.iloc[34,:]

# root.classify(test)

print(predict(root,valid1))
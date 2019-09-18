"""
rana.banik@vanderbilt.edu
check makeTree.py file for operations
"""
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tqdm
from anytree import Node, NodeMixin, RenderTree, AnyNode

def gini_index(data_obj):
    """
    :param data_obj:
    :return:
    """
    if type(data_obj)==type(pd.DataFrame()):
        data_obj = data_obj.values
    mal = np.sum(data_obj[:])
    ben = np.array(data_obj).shape[0] - mal
    tot = np.array(data_obj).shape[0]
    gini = 1 - ((mal/tot)**2+(ben/tot)**2)
    return gini

def info_gain(parent_data_obj ,left_child_obj,right_child_obj):
    """
    :param parent_data_obj:
    :param left_child_obj:
    :param right_child_obj:
    :return:
    """
    parent_gini = gini_index(parent_data_obj)
    parent_tot = parent_data_obj.shape[0]
    left_gini = gini_index(left_child_obj)
    left_tot = np.array(left_child_obj).shape[0]
    right_gini = gini_index(right_child_obj)
    right_tot = np.array(right_child_obj).shape[0]
    ig = parent_gini - ((left_tot/parent_tot)*left_gini) - ((right_tot/parent_tot)*right_gini)
    return ig

def computeOptimalSplit(parent_data):
    """
    :param parent_data: is a pandas.core.frame.DataFrame object which is a parent node
    :return: split information dictionary and child data frame nodes
    """

    metadata = {'splitting_criteria': None,
                'splitting_value': None,
                'splitting_index': [None, None],
                'samples': None,
                'left-right_split': [None, None],
                'Information_gain': None
                }

    count = 0
    IG = 0  # starting with 0 information gain
    data = parent_data.values  # (398,31) #values including label
    label = data[:, 0]  # (398,) #just the labels but can be indexed

    for feature in tqdm_notebook(range(1, data.shape[1])):
        for attribute in range(data.shape[0]):  # 0~397
            split_crit = data[attribute, feature]  # 6.981 lowest value # in feature2 39.28 is greatest
            left_df = []
            right_df = []
            for data_point in range(parent_data.shape[0]):  # 398
                count += 1  # if properly run, value should be: (#attributes x #data_points X #feature)
                if data[data_point, feature] <= split_crit:
                    left_df.append(label[data_point])
                else:
                    right_df.append(label[data_point])
            # at the end of this for loop I will have two child nodes based on chosen/first attribute
            if (np.array(left_df).shape[0] == 0 or np.array(right_df).shape[0] == 0):
                """
                If a criteria does not split at all, there should be no information gain 
                """
                split_IG = 0
            else:
                split_IG = info_gain(label, left_df, right_df)  # Now these inputs are lists of labels???

            if split_IG > IG:
                IG = split_IG
                metadata['splitting_criteria'] = parent_data.columns[feature]
                metadata['splitting_value'] = split_crit
                metadata['splitting_index'][0] = attribute
                metadata['splitting_index'][1] = feature
                metadata['samples'] = np.array(label).shape[0]
                metadata['left-right_split'][0] = np.array(left_df).shape[0]
                metadata['left-right_split'][1] = np.array(right_df).shape[0]
                metadata['Information_gain'] = IG
            else:
                continue

    left_df = pd.DataFrame()
    right_df = pd.DataFrame()

    for data_point in range(parent_data.shape[0]):  # 398
        if data[data_point, metadata['splitting_index'][1]] <= metadata['splitting_value']:
            left_df = left_df.append([parent_data.iloc[data_point]])
        else:
            right_df = right_df.append([parent_data.iloc[data_point]])
    # todo: shall I drop feature or not?
    left_df = left_df.drop(columns=metadata['splitting_criteria'])
    right_df = right_df.drop(columns=metadata['splitting_criteria'])

    return metadata, left_df, right_df

def checkLeaf(dataobject):
    """
    :param dataobject: dataframe object containing malignant and benign
    labels at the first column
    :return: logical True or False
    """
    data = np.array(dataobject)
    mal = np.sum(data[:,0])
    tot = data.shape[0]
    ben = tot - mal
    if mal == tot or ben == tot: #condition for a leaf node
        print('Leaf node')
        return True
    else:
        print('Not a leaf node')
        return False

# class Node(object):
#     def __init__(self,data,left,right,imp_feature,feat_val): #takes 5 input
#         # self.parent = None
#         self.data = data
#         self.left = left
#         self.right = right
#         self.imp_feature = imp_feature
#         self.feat_val = feat_val


# def buildTree(datafile,tree,split_info):
#     """
#     :param datafile: dataFrame object also called root node
#     :param leafNodes: empty list
#     :param split_info: empty list
#     :return:
#     leafNodes: list of leaf nodes including datapoints and features
#     split_info: list of all splitting information
#     """
#     # leafNodes = []
#     # split_info = []
#
#     # leafCount = 0
#
#     if checkLeaf(datafile):
#         # leafNodes.append(datafile)
#         node = Node(datafile, None, None, None, None)
#         tree.append(node)
#         # leafCount+=1
#         pass
#     else:
#         metadata, left, right = computeOptimalSplit(datafile)
#         split_info.append(metadata)
#         node = Node(datafile, left, right, metadata['splitting_criteria'], metadata['splitting_value'])
#         tree.append(node)
#         buildTree(left,tree,split_info)
#         buildTree(right,tree,split_info)
#
#     # print('There are total {} leaf nodes'.format(leafCount))
#
#     return tree, split_info

class Node(NodeMixin):
    def __init__(self, parent, children, name, datafile, feat_val): #name = feature name
        self.parent = None
#         self.left = left
#         self.right = right
        self.children = []
        self.name = name
        self.datafile = datafile
        self.feat_val = feat_val

class TreeNode(NodeMixin):
    def __init__(self,node_name,dataframe, feat_name,feat_val,parent = None, children=None):
        super(TreeNode,self).__init__()
        self.node_name = node_name
        self.dataframe = dataframe
        self.feat_name = feat_name
        self.feat_val = feat_val
        if parent:
            self.parent = parent
        if children:
            self.children = children

    def classify(self,datapoint):
        if len(self.children)==0:
            if (np.sum(self.dataframe.iloc[:,0])/len(self.dataframe.iloc[:,0])) >= 0.5:

                return 1
            else:
                return 0
        else:
            if datapoint[self.feat_name] <= self.feat_val: #feat
                # print(self.feat_name)
                return self.children[0].classify(datapoint) #todo
            else:
                # print(self.feat_name)
                return self.children[1].classify(datapoint)


"""
This buildTree works on Node class
"""

def buildTree(node,tree,split_info): #this is from the SplitterTry.ipynb file
    """
    :param node: class Node instance
    :param tree: list
    :param split_info:  ?? #todo
    :return:
    """
    #leafNodes = [] #will turn this again null in recursive call
    #split_info = []
    #nodeCount+=1
    if checkLeaf(node.datafile):
        #leafNodes.append(datafile)
        #nodeCount+=1
        node = Node(None, None, 'leaf', node.datafile, None)
        tree.append(node)
        #pass
    else:
        metadata, left, right = computeOptimalSplit(node.datafile)
        split_info.append(metadata)
        #nodeCount+=1
        node = Node(None,None,metadata['splitting_criteria'],node.datafile,metadata['splitting_value'])
        #tree.append(node)
        node.children = [left,right]
        buildTree(left,tree,split_info)
        buildTree(right,tree,split_info)

def buildTreeNew(node,nodeList,splitList,nodeCount):
    """
    :param data_node: Data node object, starting with a root node
    :param nodeList: []
    :param splitList: []
    :param nodeCount: starts with zero
    :return:
    """
    if checkLeaf(node.dataframe):
        nodeCount+=1 #not working
        # data_node = TreeNode(str(nodeCount),node.dataframe,None,None,parent=None,children=None)
        nodeList.append(node)
        # pass
    else:
        nodeCount+=1 #not working
        metadata, left, right = computeOptimalSplit(node.dataframe)
        # node = TreeNode(str(nodeCount),node.dataframe,metadata['splitting_criteria'],metadata['splitting_value'],parent=None,children=None)
        node.node_name=str(nodeCount)
        node.feat_name=metadata['splitting_criteria']
        node.feat_val = metadata['splitting_value']
        left_node = TreeNode("left_"+str(nodeCount),left,None,None,parent=node,children=None)
        #declaring parent = node in previous line creates one children of node. #todo: Need node.children = [left_node,right_node] ??
        right_node = TreeNode("right_"+str(nodeCount),right,None,None,parent=node,children=None)
        node.children = [left_node,right_node] #todo: is this line required?
        nodeList.append(node)
        splitList.append(metadata)
        buildTreeNew(left_node,nodeList,splitList,nodeCount)  #indentation of these two didnt work
        buildTreeNew(right_node,nodeList,splitList,nodeCount)

def predict(root,dataframe):
    correct = 0
    missed = 0
    for i in range(int(dataframe.shape[0])):
        row = dataframe.iloc[i,:]
        prediction = root.classify(row)
        if prediction == row[0]: #labels are stored in the first column
            correct+=1
        else:
            missed+=1
    return (correct/i)*100, correct


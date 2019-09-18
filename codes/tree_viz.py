import graphviz
import pydot
from IPython.display import Image,display
#
G = pydot.Dot(graph_type="digraph")
#
# node = pydot.Node(str(spinf[0]['samples']),style="filled",fillcolor="orange")
node = pydot.Node('Demo_node')
#
G.add_node(node)
# #
im = Image(G.create())
# #
display(im)
# #
#
# from graphviz import Graph
#
# e = Graph('ER', filename='er.gv', engine='neato')
#
# e.attr('node', shape='box')
# e.node('course')
# e.node('institute')
# e.node('student')
#
# e.attr('node', shape='ellipse')
# e.node('name0', label='name')
# e.node('name1', label='name')
# e.node('name2', label='name')
# e.node('code')
# e.node('grade')
# e.node('number')
#
# e.attr('node', shape='diamond', style='filled', color='lightgrey')
# e.node('C-I')
# e.node('S-C')
# e.node('S-I')
#
# e.edge('name0', 'course')
# e.edge('code', 'course')
# e.edge('course', 'C-I', label='n', len='1.00')
# e.edge('C-I', 'institute', label='1', len='1.00')
# e.edge('institute', 'name1')
# e.edge('institute', 'S-I', label='1', len='1.00')
# e.edge('S-I', 'student', label='n', len='1.00')
# e.edge('student', 'grade')
# e.edge('student', 'name2')
# e.edge('student', 'number')
# e.edge('student', 'S-C', label='m', len='1.00')
# e.edge('S-C', 'course', label='n', len='1.00')
#
# e.attr(label=r'\n\nEntity Relation Diagram\ndrawn by NEATO')
# e.attr(fontsize='20')
#
# e.view(filename='Creat')
#
# from graphviz import Digraph
#
# g = Digraph('G', filename='hello.gv')
#
# g.edge('Hello', 'World')
#
# g.view()



# Make a prediction with a decision tree
# def predict(node, row):
# 	if row[node['index']] < node['value']:
# 		if isinstance(node['left'], dict):
# 			return predict(node['left'], row)
# 		else:
# 			return node['left']
# 	else:
# 		if isinstance(node['right'], dict):
# 			return predict(node['right'], row)
# 		else:
# 			return node['right']
#
#
# # stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
#
# dataset = [[2.771244718,1.784783929,0],
# 	[1.728571309,1.169761413,0],
# 	[3.678319846,2.81281357,0],
# 	[3.961043357,2.61995032,0],
# 	[2.999208922,2.209014212,0],
# 	[7.497545867,3.162953546,1],
# 	[9.00220326,3.339047188,1],
# 	[7.444542326,0.476683375,1],
# 	[10.12493903,3.234550982,1],
# 	[6.642287351,3.319983761,1]]
#
# #  predict with a stump
# stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
# for row in dataset:
# 	prediction = predict(stump, row)
# 	print('Expected=%d, Got=%d' % (row[-1], prediction))
"""
this file is from Azhar
"""

class Node(object):
    def __init__(self, info):
        self.info = info
        self.left = None
        self.right = None


def is_pure(node):
    return node.info > 7


def build_recurse_tree(node):
    if is_pure(node):
        print('node is pure')
    else:
        # meta, node.left, node.right = computeOptimalSplit(node)
        node.left = build_recurse_tree(Node(2 * node.info))
        node.right = build_recurse_tree(Node(2 * node.info + 1))
        return node

def inorderTraverse(node):
    print(node.info)
#     print(node.left)
    if node.left!=None:
        inorderTraverse(node.left)
    if node.right!=None:
        inorderTraverse(node.right)

root = Node(1)
root = build_recurse_tree(root)
inorderTraverse(root)





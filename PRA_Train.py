# coding=utf-8
'''
为PRA算法训练逻辑回归模型
'''
#import tensorflow as tf
import numpy as np
from py2neo import Graph 
from itertools import combinations,permutations


Military = Graph('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='neo4j')
#relations = Military.call.db.relationshipTypes()
    #print(Military.apoc.rel.startNode(record))
#print(dir(Military.call))
class PRA_Model():
    """
    PRA算法模型类
    """
    def __init__(self,Graph,relation_num=2,predicted_relationship=['belong','Person','Country']):
        """
        初始化
        predicted_relationship 为列表[预测的关系名，头节点，尾节点]
        relation_num 为限定的关系路径包含的关系数量
        Graph 为py2neo GRAPH对象
        """
        self.predicted_relation = predicted_relationship
        self.relation_num = relation_num
        self.Graph = Graph

    def find_paths(self,mode):
        """
        找出符合关系的所有路径 
        mode即寻找符合关系路径的模式,permutions为全排列模式
        """
        new_path = []
        if mode == 'permutions':
            relations = self.Graph.call.db.relationshipTypes().data()#返回所有关系类型,列表返回
            paths = list(permutations(relations,self.relation_num))#对关系全排列
            for path in paths:
                if (self.Graph.evaluate( 'MATCH ()-[r:'+path[0]['relationshipType']+']->() RETURN labels(startNode(r))')[0] == self.predicted_relation[1]) and (self.Graph.evaluate( 'MATCH ()-[r:'+path[-1]['relationshipType']+']->() RETURN labels(endNode(r))')[0] == self.predicted_relation[2]):
                    new_path.append(path)
            return new_path
        else:
            print('Please input the mode')
            #print(new_path)
            #print('new_path'+str(len(new_path)))

    def compute_feature(self,path,fromnode,tonode,direction=1):
        """
        计算路径特征值,后向截断的策略
        path是关系路径一个包含关系的元组，例:({'relationship':'HAS'},{'relationship':'IN_COUNTRY'},{'relationship':'IN_LEAGUE'})
        fromnode是路径的头节点:例{'label':'Person','name':'liqiang'}或id?
        tonode是路径的尾节点  :例{'label':'Person','name':'hanmeimei'}或id?
        """
        #Prl = 0 #probability of reaching node e from e' with a one step random walk with relationship type Rl
        #nodee_1 = 0 #e'
        #nodee   = 0 #e
        #range_P_1 = 0 #range of P'
        length = len(path) #length of path
        hspe = 1 #the path feature value
        if direction == 0:#注意正向迭代迭代
            if len(path)==1:
                query = "MATCH (N),(M)-[r:"+path[0]['relationship']+"]->(tar)"
                print()
                if self.Graph.run(query).data()[0]=="true":
                    hspe = 1/n
                else:
                    hspe = 0
                return hspe
            #else:

            #for relation in path:
            #    #首先找到本次关系的尾节点,py2neo.database.work.Cursor
            #    Endnodes = self.Graph.run('MATCH ()-[r:'+relation['relationship']+']->() RETURN endNode(r)')
            #    for endnode in Endnodes:#endnode为record类型,endnode[0]即为node类型
            #        print(endnode[0])
            #        print(type(endnode[0]))
            #        break
            #    break
        elif direction == 1:#注意反向迭代
            if length == 1:#关系路径长度只剩1
                query = "MATCH (N),(M)-[r:"+path[0]['relationship']+"]->(tar) WHERE id(N)="+str(tonode)+"AND id(M)="+str(fromnode)+"WITH N,[tar] AS target RETURN N IN target"
                if self.Graph.run(query).data()[0]['N IN target']==True:#如果到节点与从节点通过关系链接的话
                    hspe = 1/
                else:
                    hspe = 0 
        return hspe

    def train(self):
        """训练"""

    def predict(self):
        """预测"""

if __name__=="__main__":
    PRA = PRA_Model(relation_num=3,Graph=Military)
    #PRA.find_paths(mode='permutions')
    PRA.compute_feature(path=[{'relationship':'HAS'}],fromnode=125,tonode=131)

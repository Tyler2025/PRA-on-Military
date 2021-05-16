# coding=utf-8
'''
为PRA算法训练逻辑回归模型
'''
import numpy as np
import pandas as pd
from py2neo import Graph 
from itertools import combinations,permutations
from collections import OrderedDict
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import time

Military = Graph('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='neo4j')

class PRA_Model():
    """
    PRA算法模型类
    """
    def __init__(self,Graph,relation_num=2,predicted_relationship=['actedin','Actor','Genre']):
        """
        初始化
        predicted_relationship 为列表[预测的关系名，头节点，尾节点]
        relation_num 为限定的关系路径包含的关系数量
        Graph 为py2neo GRAPH对象
        """
        self.predicted_relation = predicted_relationship
        self.relation_num = relation_num
        self.Graph = Graph
        self.Model = 0
    def find_paths(self,mode):
        """
        找出符合关系的所有路径 
        mode即寻找符合关系路径的模式,permutions为全排列模式
        """
        new_path = []
        paths = []
        if mode == 'permutions':
            relations = self.Graph.call.db.relationshipTypes().data()#返回所有关系类型,列表返回
            per_num = 2
            while per_num <= self.relation_num:
                paths = paths+list(permutations(relations,per_num))
                per_num += 1
            for path in paths:
                if (self.predicted_relation[1] in self.Graph.evaluate( 'MATCH ()-[r:'+path[0]['relationshipType']+']->() RETURN labels(startNode(r))')) and (self.predicted_relation[2] in self.Graph.evaluate( 'MATCH ()-[r:'+path[-1]['relationshipType']+']->() RETURN labels(endNode(r))')):
                    new_path.append(path)
            return new_path
        else:
            print('Please input the mode')


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
        hspe = 0 #the path feature value
        if direction == 0:#注意正向迭代迭代
            if len(path)==1:
                query = "MATCH (N),(M)-[r:"+path[0]['relationshipType']+"]->(tar)"
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
                query = "MATCH (N),(M)-[r:"+path[0]['relationshipType']+"]->(tar) WHERE id(N)="+str(tonode)+"AND id(M)="+str(fromnode)+"WITH N,collect(tar) AS target RETURN N IN target"
                data = self.Graph.run(query).data()
                if not data:
                    hspe = 0
                elif data[0]['N IN target']==True:#如果到节点与从节点通过关系链接的话
                    query_end = "MATCH (M)-[r:"+path[0]['relationshipType']+"]->(N) WHERE id(M)="+str(fromnode)+"RETURN count(N)"
                    hspe = 1/self.Graph.run(query_end).data()[0]['count(N)']
                else:
                    hspe = 0
                return hspe
            else:#关系路径长度大于1
                query_start = "MATCH (M)-[r:"+path[0]['relationshipType']+"]->(N) WHERE id(M)="+str(fromnode)+"RETURN count(N)"
                nodes_count = self.Graph.run(query_start).data()[0]['count(N)']#首先找出从节点通过关系0链接的节点数
                if nodes_count == 0:
                    return 0
                else:
                    query_middle = "MATCH (M)-[r:"+path[0]['relationshipType']+"]->(N) WHERE id(M)="+str(fromnode)+"RETURN id(N)"
                    nodes = self.Graph.run(query_middle).data()#找出链接的节点
                    for entity in nodes:
                        hspe += (1/nodes_count) * self.compute_feature(path[1:],entity['id(N)'],tonode)
                    return hspe
        return hspe

    def data_construt(self,paths):
        """构建训练集与测试集数据"""
        querysids = "MATCH (m:"+self.predicted_relation[1]+") WITH id(m) AS sid LIMIT 10 RETURN collect(sid) as sids"#找出路径首位节点id
        sids = self.Graph.run(querysids).data()[0]['sids']
        querytids = "MATCH (m:"+self.predicted_relation[2]+") WITH id(m) AS tid LIMIT 10 RETURN collect(tid) as tids"
        tids = self.Graph.run(querytids).data()[0]['tids']
        node_pairs = [0]*len(sids)*len(tids)
        i = 0
        for sid in sids:#得到所有的节点对组合
            for tid in tids:
                node_pairs[i] = [sid,tid]
                i += 1
        i = 0
        feature_matrice = [0]*len(paths)
        node_pair_feature = []
        labels = []
        data = {}
        t1 = time.clock()
        for path in paths:#计算对应节点对的路径特征向量
            for node_pair in node_pairs:
                path_feature = self.compute_feature(path,node_pair[0],node_pair[1])#计算路径特征值
                if i == 0:
                    if path_feature == 0:#判断关系标签,只运行一遍
                        labels = labels + [0]
                    else:
                        labels = labels + [1]
                node_pair_feature = node_pair_feature + [path_feature]
            feature_matrice[i] = node_pair_feature
            data['path'+str(i)] = node_pair_feature
            node_pair_feature = []
            #print('process time spent:',time.clock()-t1)
            i += 1        
        data['relation_labels'] = labels
        train_data = pd.DataFrame(data)#转换为DataFrame
        print('训练数据格式:')
        print(train_data.head())
        i = 0
        exam_X = train_data.loc[:,['path'+str(i) for i in range(len(paths))]]#提取特征向量与标签
        exam_Y = train_data.loc[:,'relation_labels']
        X_train,X_test,Y_train,Y_test=train_test_split(exam_X,exam_Y,train_size= .8)#分开训练集与测试集数据
        #X_train=X_train.values.reshape(-1,1)
        #X_test=X_test.values.reshape(-1,1)
        return X_train,X_test,Y_train,Y_test

    def train(self,X_train,Y_train,X_test,Y_test):
        """训练"""
        print('X_train:',X_train.shape)
        print('Y_train:',Y_train.shape)
        print('X_test:',X_test.shape)
        print('Y_test:',Y_test.shape)
        model = LogisticRegression(verbose=1)#实例化逻辑回归分类器
        model.fit(X_train,Y_train)
        self.Model = model
        print('准确率:'+str(model.score(X_test,Y_test)))
        joblib.dump(model,'pra.model')#保存模型为pra.model
        print('Model has been stored')

    def predict(self,fromnode,tonode,paths,model):
        """预测"""
        self.Model = joblib.load(model)
        print('Model has been loaded')
        pre_data = {}
        i = 0
        for path in paths:
            pre_data['path'+str(i)] = self.compute_feature(path,fromnode,tonode)
            i += 1
        pre_data = pd.Series(pre_data).values.reshape(1,-1)
        print('pre_data:',pre_data,' shape:',pre_data.shape)
        print(self.Model.predict_proba(pre_data))
        print(self.Model.predict(pre_data))
        print('Weights:',self.Model.coef_,'Bias:',self.Model.intercept_)

if __name__=="__main__":
    PRA = PRA_Model(relation_num=3,Graph=Military)
    #paths = PRA.find_paths(mode='permutions')
    #print(paths)
    paths = [({'relationshipType': 'ACTED_IN'}, {'relationshipType': 'IN_GENRE'}), ({'relationshipType': 'ACTED_IN'}, {'relationshipType': 'RATED'}, {'relationshipType': 'IN_GENRE'}), ({'relationshipType': 'ACTED_IN'}, {'relationshipType': 'DIRECTED'}, {'relationshipType': 'IN_GENRE'})]
    #X_train,X_test,Y_train,Y_test = PRA.data_construt(paths)
    #PRA.train(X_train,Y_train,X_test,Y_test)
    PRA.predict(9848,4,paths,'pra.model')
    #print('path_feature:',PRA.compute_feature(path = paths[0],fromnode=9836,tonode=6))

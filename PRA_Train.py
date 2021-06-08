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
import pickle
import time
import math

Military = Graph('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='military66')

class PRA_Model():
    """
    PRA算法模型类
    """
    def __init__(self,Graph,predicted_relationship=['actedin','生产单位','产国']):
        """
        初始化
        predicted_relationship 为列表[预测的关系名，头节点，尾节点]
        Graph 为py2neo GRAPH对象
        """
        self.predicted_relation = predicted_relationship
        self.Graph = Graph
        self.Model = 0

    def find_paths(self,mode,config):
        """
        找出符合关系的所有路径 
        mode即寻找符合关系路径的模式,permutions为全排列模式,randomwalk为随机游走模式,BFS为广度优先搜索策略
        randomwalk：length：限制单边最大关系路径长度，example_num：指定单边使用的节点数量，steps：随机游走数量
        """
        new_path = []
        paths = []
        if mode == 'permutions':
            relations = self.Graph.call.db.relationshipTypes().data()#返回所有关系类型,列表返回
            per_num = 2
            while per_num <= config['relation_num']:
                paths = paths+list(permutations(relations,per_num))
                per_num += 1
            for path in paths:
                if (self.predicted_relation[1] in self.Graph.evaluate( 'MATCH ()-[r:'+path[0]['relationshipType']+']->() RETURN labels(startNode(r))')) and (self.predicted_relation[2] in self.Graph.evaluate( 'MATCH ()-[r:'+path[-1]['relationshipType']+']->() RETURN labels(endNode(r))')):
                    new_path.append(path)
            return new_path
        elif mode == 'randomwalk':
            #query_randomwalk = "MATCH (startNode:"+self.predicted_relationship[1]+") CALL gds.alpha.randomWalk.stream({nodeProjection: '*',relationshipProjection: '*',start: id(startNode),steps: "+3+",walks:"+ config[steps]+"})"
            sub_graph_start = []#子图特征
            sub_graph_target = []
            potential_path = []
            for i in range(1,config['length']+1):#开始随机游走，先是源节点，再是目标节点
                query_randomwalk_start = "MATCH (startNode:"+self.predicted_relation[1]+") WITH startNode AS startNodes LIMIT "+str(config['example_num'])+" CALL gds.alpha.randomWalk.stream({nodeProjection: '*',relationshipProjection: '*',start: id(startNodes),steps: "+str(i)+",walks:"+ str(config['steps'])+"}) YIELD nodeIds RETURN DISTINCT nodeIds"
                temp_s = self.Graph.run(query_randomwalk_start).data()
                for temp in temp_s:
                    sub_graph_start.append(temp['nodeIds'])
                    sub_graph_start.append([temp['nodeIds'][0]])#还得加上目的节点的ID,可以搜索直接走到目的节点的路径
                query_randomwalk_target = "MATCH (startNode:"+self.predicted_relation[2]+") WITH startNode AS startNodes LIMIT "+str(config['example_num'])+" CALL gds.alpha.randomWalk.stream({nodeProjection: '*',relationshipProjection: '*',start: id(startNodes),steps: "+str(i)+",walks:"+ str(config['steps'])+"}) YIELD nodeIds RETURN DISTINCT nodeIds"
                temp_t = self.Graph.run(query_randomwalk_target).data()
                for temp in temp_t:
                    sub_graph_target.append(temp['nodeIds'])#注意关系的方向问题
                    sub_graph_target.append([temp['nodeIds'][0]])#还得加上目的节点的ID,可以搜索直接走到目的节点的路径
            sub_graph_start = [list(t) for t in set(tuple(_) for _ in sub_graph_start)]
            sub_graph_target = [list(t) for t in set(tuple(_) for _ in sub_graph_target)]#列表中去除相同的列表元素

            #开始融合子图特征以生成潜在路径
            for target_intermedia in sub_graph_target:
                for source_intermedia in sub_graph_start:
                    if target_intermedia[-1] == source_intermedia[-1]:
                        temp = source_intermedia[:-1] + list(reversed(target_intermedia))
                        potential_path.append(temp)
            #找出关系类型，并通过关系类型出现的次数来遴选关系路径
            paths = self.parse_potential(potential_path)
            return paths
            #print('Job')
        elif mode=="BFS":#注意方向问题，需要嵌套写法吗？         
            time_start = time.time() #开始计时
            #query_BFS = "MATCH (startNode:Actor) WITH startNode AS startNodes LIMIT 10 CALL gds.alpha.bfs.stream({nodeProjection: '*',relationshipProjection: '*',startNode: id(startNodes),maxDepth:1}) YIELD nodeIds RETURN nodeIds,id(startNodes) AS more_info"
            sub_graph_start = []
            sub_graph_target = []
            potential_path = []
            for i in range(config['length']):
                print('i=',i)
                if i == 0:
                    query_BFS_start = "MATCH (startNode:"+self.predicted_relation[1]+") WITH startNode AS startNodes LIMIT "+str(config['example_num'])+" CALL gds.alpha.bfs.stream({nodeProjection: '*',relationshipProjection: '*',startNode: id(startNodes),maxDepth:1}) YIELD nodeIds RETURN nodeIds"
                    temp_s = self.Graph.run(query_BFS_start).data()
                    for temp in temp_s:
                        sub_graph_start.append([temp['nodeIds'][0]])
                        nodes_num = len(temp['nodeIds'])
                        if nodes_num > 2 and nodes_num < config['constraint']:#需要改进，抛弃出度大于阈值的节点,不能回头
                            for inter_node in temp['nodeIds'][1:]:
                                sub_graph_start.append([temp['nodeIds'][0]]+[inter_node])
                        elif nodes_num >= config['constraint']:
                            for inter_node in temp['nodeIds'][1:config['constraint']+1]:
                                sub_graph_start.append([temp['nodeIds'][0]]+[inter_node])
                        else:
                            sub_graph_start.append(temp['nodeIds'])
                    query_BFS_target = "MATCH (startNode:"+self.predicted_relation[2]+") WITH startNode AS startNodes LIMIT "+str(config['example_num'])+" CALL gds.alpha.bfs.stream({nodeProjection: '*',relationshipProjection: '*',startNode: id(startNodes),maxDepth:1}) YIELD nodeIds RETURN nodeIds"
                    temp_t = self.Graph.run(query_BFS_target).data()
                    for temp in temp_t:
                        sub_graph_target.append([temp['nodeIds'][0]])
                        nodes_num = len(temp['nodeIds'])
                        if nodes_num > 2 and nodes_num < config['constraint']:#需要改进，抛弃出度大于阈值的节点,不能回头
                            for inter_node in temp['nodeIds'][1:]:
                                sub_graph_target.append([temp['nodeIds'][0]]+[inter_node])
                        elif nodes_num >= config['constraint']:
                            for inter_node in temp['nodeIds'][1:config['constraint']+1]:
                                sub_graph_target.append([temp['nodeIds'][0]]+[inter_node])
                        else:
                            sub_graph_target.append(temp['nodeIds'])
                else:
                    for inter_path in sub_graph_start:
                        if len(inter_path) == i+1:
                            query_BFS_inter_start = "MATCH (start) WHERE id(start)="+str(inter_path[-1])+" CALL gds.alpha.bfs.stream({nodeProjection: '*',relationshipProjection: '*',startNode: id(start),maxDepth:1}) YIELD nodeIds RETURN nodeIds"
                            interr_nodes = self.Graph.run(query_BFS_inter_start).data()[0]['nodeIds'][1:]
                            if len(interr_nodes) > config['constraint']:#限制节点出度
                                interr_nodes = interr_nodes[:config['constraint']]
                            for inter in interr_nodes:
                                if inter not in inter_path:
                                    inter_temp_s = inter_path + [inter]
                                    sub_graph_start.append(inter_temp_s)
                    for inter_path in sub_graph_target:
                        if len(inter_path) == i+1:
                            query_BFS_inter_start = "MATCH (start) WHERE id(start)="+str(inter_path[-1])+" CALL gds.alpha.bfs.stream({nodeProjection: '*',relationshipProjection: '*',startNode: id(start),maxDepth:1}) YIELD nodeIds RETURN nodeIds"
                            interr_nodes = self.Graph.run(query_BFS_inter_start).data()[0]['nodeIds'][1:]
                            if len(interr_nodes) > config['constraint']:
                                interr_nodes = interr_nodes[:config['constraint']]
                            for inter in interr_nodes:
                                if inter not in inter_path:#不能回头
                                    inter_temp_s = inter_path + [inter]
                                    sub_graph_target.append(inter_temp_s)
            time_end1 = time.time()    #结束计时
            time_c= time_end1 - time_start   #运行所花时间
            print('subgraph feature time cost:', time_c, 's',' sub_graph_start_num:',len(sub_graph_start),' sub_graph_target_num:',len(sub_graph_target))
            #首先，采样
            if 'strategy' in config.keys():
                sub_graph_start = self.sample_potential_paths(sub_graph_start,config['sample_num'],config['strategy'],2*config['length'])
                sub_graph_target = self.sample_potential_paths(sub_graph_target,config['sample_num'],config['strategy'],2*config['length'])
            #开始融合子图特征以生成潜在路径
            potential_path = self.connect_sub_graph(sub_graph_start,sub_graph_target)
            #考虑添加限制，现阶段生成80多万条潜在路径，实在太多
            time_end2 = time.time()    #结束计时
            time_c= time_end2 - time_end1   #运行所花时间
            print('concentrate time cost:', time_c, 's','potential_path:',len(potential_path))

            #找出关系类型，并通过关系类型出现的次数来遴选关系路径
            paths = self.parse_potential(potential_path)

            time_end = time.time()    #结束计时
            time_c= time_end - time_start   #运行所花时间
            print('time cost', time_c, 's','ultimate_paths:',len(paths))
            print('Path Feature Selected By SFE')
            return paths
        else:
            print('Please input the mode')

    def connect_sub_graph(self,sub_graph_start,sub_graph_target):
        """连接两个子图特征"""
        potential_path=[]
        for target_intermedia in sub_graph_target:
            for source_intermedia in sub_graph_start:
                if target_intermedia[-1] == source_intermedia[-1]:
                    temp = source_intermedia[:-1] + list(reversed(target_intermedia))
                    potential_path.append(temp)
        return potential_path

    def sample_potential_paths(self,potential_path,sample_num,strategy,max_len):
        """对潜在路径抽样"""
        samples = []
        if strategy == 'random': 
            df = pd.DataFrame({'potential_path':pd.Series(potential_path)})
            #print(df.head())
            df_samples = df.sample(n=sample_num,replace=False,random_state=0)
            #print(df_samples.head())
            samples = df_samples['potential_path'].values.tolist()
        elif strategy == 'stratified':
            stratified_length = [0]*max_len
            for path in potential_path:
                index = len(path)-2
                if stratified_length[index] == 0:
                    stratified_length[index]=[path]
                else:
                    stratified_length[index].append(path)
            for i in range(stratified_length.count(0)):
                stratified_length.remove(0)#去除没有该种长度的路径
            quarter = math.floor(sample_num/len(stratified_length))
            for dump in stratified_length:
                if quarter < len(dump):
                    df = pd.DataFrame({'potential_path':pd.Series(dump)})
                    df_samples = df.sample(n=quarter,replace=False,random_state=0)#注意当采样的样本总数小于采样数目时会抛出错误
                    samples = samples + df_samples['potential_path'].values.tolist()
                else:
                    samples = samples + dump#注意一下
        return samples

    def parse_potential(self,paths):
        """解析潜在路径返回关系类型序列"""
        potential_reltypes = []
        selected_path = []
        for path in paths:
            query_head = "MATCH "
            query_where = "WHERE "
            query_unwind = "UNWIND ["
            query_return = " RETURN type(rela) AS relationshipType"
            rel_typess = []
            for i in range(len(path)):
                query_head = query_head + "(m"+str(i)+")-[r"+str(i)+"]->"#注意这是无方向查询###
                query_where = query_where + "id(m" + str(i) + ")=" + str(path[i]) + " AND "
                query_unwind = query_unwind + "r" + str(i) +","
            query_head = query_head[:-7]
            query_where = query_where[:-5]
            query_unwind = query_unwind[:-4]+'] AS rela'
            query = query_head + query_where + query_unwind + query_return
            rel_types = self.Graph.run(query).data()#得到关系类型列表
            for rel_type in rel_types:
                rel_typess.append(rel_type['relationshipType'])
            if len(rel_types) == len(path)-1:
                potential_reltypes.append(tuple(rel_typess))
            else:
                for rel_seq in  self.list_split(rel_typess,(len(path)-1)):
                    potential_reltypes.append(tuple(rel_seq))
        potential_reltypes = list(set(potential_reltypes))#这里可以添加过滤的方法，比如Lao的acc方法，Mardner的常见次数方法
        for potential_reltype in potential_reltypes:
            temp = []
            for rep in potential_reltype:
                temp = temp + [{'relationshipType':rep}]
            selected_path = selected_path + [tuple(temp)]
        #print('selected')
        return selected_path

    def list_split(self,listTemp, n):
        '''列表均分函数，返回迭代器'''
        for i in range(0, len(listTemp),n):
            yield listTemp[i:i + n]

    def compute_feature(self,path,fromnode,tonode):
        """
        计算路径特征值,后向截断的策略
        path是关系路径一个包含关系的元组，例:({'relationship':'HAS'},{'relationship':'IN_COUNTRY'},{'relationship':'IN_LEAGUE'})
        fromnode是路径的头节点:例{'label':'Person','name':'liqiang'}或id?
        tonode是路径的尾节点  :例{'label':'Person','name':'hanmeimei'}或id?
        """
        length = len(path) #length of path
        hspe = 0 #the path feature value
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

    def data_construt(self,paths,scale,config,mode):
        """构建训练集与测试集数据"""
        #querysids = "MATCH (m:"+self.predicted_relation[1]+") WITH id(m) AS sid LIMIT 10 RETURN collect(sid) as sids"#找出路径首位节点id
        #sids = self.Graph.run(querysids).data()[0]['sids']
        #querytids = "MATCH (m:"+self.predicted_relation[2]+") WITH id(m) AS tid LIMIT 10 RETURN collect(tid) as tids"
        #tids = self.Graph.run(querytids).data()[0]['tids']
        if mode == 0:
            labels_qualified = self.load_labels('Labels.csv','labels.txt',1)
            querysids = "MATCH (m:"+self.predicted_relation[1]+") WHERE id(m)<>1400 WITH id(m) AS sid LIMIT "+str(scale)+" RETURN collect(sid) as sids"#找出路径首位节点id
            sids = self.Graph.run(querysids).data()[0]['sids']
            querytids = "MATCH (m:"+self.predicted_relation[2]+") WITH id(m) AS tid LIMIT "+str(scale)+" RETURN collect(tid) as tids"
            tids = self.Graph.run(querytids).data()[0]['tids']
            node_pairs = [0]*len(sids)*len(tids)
            i = 0
            for sid in sids:#得到所有的节点对组合
                for tid in tids:
                    node_pairs[i] = [sid,tid]
                    i += 1
            i = 0
            node_pair_feature = []
            labels = []
            data = {}
            for path in paths:#计算对应节点对的路径特征向量
                time_start = time.time() #开始计时
                for node_pair in node_pairs:
                    path_feature = self.compute_feature(path,node_pair[0],node_pair[1])#计算路径特征值
                    if i == 0:
                        if node_pair in labels_qualified:
                            labels = labels + [1]
                        else:
                            labels = labels + [0]
                    node_pair_feature = node_pair_feature + [path_feature]
                data['path'+str(i)] = node_pair_feature
                time_end = time.time() #开始计时
                print('Path',i,'Computed,','Time Cost:',time_end-time_start,'s')
                node_pair_feature = [] 
                i += 1  

            data['relation_labels'] = labels
            with open(config['data_name'],'wb') as f:
                pickle.dump(data,f)

        #feature_matrice = [0]*len(paths)
        #node_pair_feature = []
        #labels = []
        #data = {}
        #t1 = time.clock()
        #for path in paths:#计算对应节点对的路径特征向量
        #    for node_pair in node_pairs:
        #        path_feature = self.compute_feature(path,node_pair[0],node_pair[1])#计算路径特征值
        #        if i == 0:
        #            if path_feature == 0:#判断关系标签,只运行一遍
        #                labels = labels + [0]
        #            else:
        #                labels = labels + [1]
        #        node_pair_feature = node_pair_feature + [path_feature]
        #    feature_matrice[i] = node_pair_feature
        #    data['path'+str(i)] = node_pair_feature
        #    node_pair_feature = []
        #    #print('process time spent:',time.clock()-t1)
        #    i += 1        
        elif mode == 1:
            with open(config['data_name'],'rb') as f:
                data = pickle.load(f)
        train_data = pd.DataFrame(data)#转换为DataFrame
        print('训练数据格式:')
        print(train_data.head())
        i = 0
        exam_X = train_data.loc[:,['path'+str(i) for i in range(len(paths))]]#提取特征向量与标签
        exam_Y = train_data.loc[:,'relation_labels']
        X_train,X_test,Y_train,Y_test=train_test_split(exam_X,exam_Y,train_size= 0.7)#分开训练集与测试集数据
        return X_train,X_test,Y_train,Y_test

    def sfe_data_construct(self,paths,scale,main_col,config):
        """
        训练集与测试集构建
        """
        querysids = "MATCH (m:"+self.predicted_relation[1]+") WHERE id(m)<>1400 WITH id(m) AS sid LIMIT "+str(scale)+" RETURN collect(sid) as sids"#找出路径首位节点id
        sids = self.Graph.run(querysids).data()[0]['sids']
        querytids = "MATCH (m:"+self.predicted_relation[2]+") WITH id(m) AS tid LIMIT "+str(scale)+" RETURN collect(tid) as tids"
        tids = self.Graph.run(querytids).data()[0]['tids']
        node_pairs = [0]*len(sids)*len(tids)
        i = 0
        for sid in sids:#得到所有的节点对组合
            for tid in tids:
                node_pairs[i] = [sid,tid]
                i += 1
        i = 0
        node_pair_feature = []
        labels = []
        data = {}
        for path in paths:#计算对应节点对的路径特征向量
            time_start = time.time() #开始计时
            for node_pair in node_pairs:
                path_feature = self.sfe_compute_feature(path,node_pair[0],node_pair[1])#计算路径特征值
                if i == main_col:
                    if path_feature == 0:#判断关系标签,只运行一遍
                        labels = labels + [0]
                    else:
                        labels = labels + [1]
                node_pair_feature = node_pair_feature + [path_feature]
            data['path'+str(i)] = node_pair_feature
            time_end = time.time() #开始计时
            print('Path',i,'Computed,','Time Cost:',time_end-time_start,'s')
            node_pair_feature = [] 
            i += 1  

        data['relation_labels'] = labels
        with open(config['data_name'],'wb') as f:
            pickle.dump(data,f)

        #with open('dataset.txt','rb') as f:
        #    data = pickle.load(f)

        train_data = pd.DataFrame(data)#转换为DataFrame
        print('训练数据格式:')
        print(train_data.head())
        i = 0
        exam_X = train_data.loc[:,['path'+str(i) for i in range(len(paths))]]#提取特征向量与标签
        exam_Y = train_data.loc[:,'relation_labels']
        X_train,X_test,Y_train,Y_test=train_test_split(exam_X,exam_Y,train_size= 0.8)#分开训练集与测试集数据
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
    #paths = PRA.find_paths(mode='randomwalk',config={'length':2,'steps':3,'example_num':10})
    paths = PRA.find_paths(mode='BFS',config={'length':3,'example_num':10,'constraint':20,'sample_num':1000,'strategy':'stratified'})
    for path in paths:
        print(path)

    #paths = PRA.find_paths(mode='permutions')
    #print(paths)
    #paths = [({'relationshipType': 'ACTED_IN'}, {'relationshipType': 'IN_GENRE'}), ({'relationshipType': 'ACTED_IN'}, {'relationshipType': 'RATED'}, {'relationshipType': 'IN_GENRE'}), ({'relationshipType': 'ACTED_IN'}, {'relationshipType': 'DIRECTED'}, {'relationshipType': 'IN_GENRE'})]
    #X_train,X_test,Y_train,Y_test = PRA.data_construt(paths)
    #PRA.train(X_train,Y_train,X_test,Y_test)
    #PRA.predict(9848,4,paths,'pra.model')
    #print('path_feature:',PRA.compute_feature(path = paths[0],fromnode=9836,tonode=6))

    #paths = [[4,1,562,351],[2,4568,19502],[365,251],[1598,256],[12,3498,2456],[12,32],[12,2]]
    #PRA.sample_potential_paths(paths,6,'stratified',6)

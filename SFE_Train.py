# coding=utf-8
'''
为SFE算法训练逻辑回归模型
'''
from PRA_Train import PRA_Model
from py2neo import Graph 
import pickle
import time 
import pandas as pd 
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import csv

class SFE_Model(PRA_Model):
    """
    SFE算法模型类,继承自PRA
    """
    def __init__(self, Graph,predicted_relationship,config=None):
        """
        初始化SFE类
        """
        super(SFE_Model,self).__init__(Graph,predicted_relationship)
        self.config = config

    def sfe_compute_feature(self,path,fromnode,tonode):
        """
        sfe算法计算路径特征值的方法
        path:包含一系列关系类型的列表，例：({'relationship':'HAS'},{'relationship':'IN_COUNTRY'},{'relationship':'IN_LEAGUE'})
        fromnode：从节点id，tonode：到节点id
        """
        query_head = "MATCH (fromnode),(tonode)"
        query_middle = ""
        for relation in path:
            query_middle = query_middle+"-[:"+relation['relationshipType']+"]->()"
        query_middle = query_middle[:-1]+"tonode)"
        query_where = " WHERE id(fromnode)="+str(fromnode)+" AND id(tonode)="+str(tonode)
        query_exist = " RETURN exists((fromnode)"+query_middle+") AS Feature"
        query = query_head  + query_where + query_exist
        try:
            result = self.Graph.run(query).data()[0]['Feature']
        except:#异常
            print('Exception:something going wrong,perform reseult = 0')
            result = 0
        return int(result)

    def load_labels(self,label_file,label_name,rw):
        """
        加载标签文件
        """
        labels = []
        if rw == 0:
            with open(label_file,newline='',encoding='utf-8') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for i,row in enumerate(csvreader):
                        if i > 0:
                            labels.append([int(row[0]),int(row[3])])
            with open(label_name,'wb') as f:
                pickle.dump(labels,f)
            return labels
        else:
            with open(label_name,'rb') as f:
                labels = pickle.load(f)
                return labels

    def sfe_data_construct(self,paths,scale,main_col,config):
        """
        训练集与测试集构建
        """
        labels_qualified = SFE.load_labels('Labels.csv','labels.txt',1)
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
                #if i == main_col:
                #    if path_feature == 0:#判断关系标签,只运行一遍
                #        labels = labels + [0]
                #    else:
                #        labels = labels + [1]
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

        #with open(config['data_name'],'rb') as f:
        #    data = pickle.load(f)

        train_data = pd.DataFrame(data)#转换为DataFrame
        print('训练数据格式:')
        print(train_data.head())
        i = 0
        exam_X = train_data.loc[:,['path'+str(i) for i in range(len(paths))]]#提取特征向量与标签
        exam_Y = train_data.loc[:,'relation_labels']
        X_train,X_test,Y_train,Y_test=train_test_split(exam_X,exam_Y,train_size= 0.8)#分开训练集与测试集数据
        return X_train,X_test,Y_train,Y_test

    def sfe_train(self,X_train,Y_train,X_test,Y_test,config):
        """SFE训练"""
        print('X_train:',X_train.shape)
        print('Y_train:',Y_train.shape)
        print('X_test:',X_test.shape)
        print('Y_test:',Y_test.shape)
        model = LogisticRegression(verbose=1)#实例化逻辑回归分类器
        model.fit(X_train,Y_train)
        self.Model = model
        print('准确率:'+str(model.score(X_test,Y_test)))
        joblib.dump(model,config['model_name'])#保存模型为sfe.model
        print('Model has been stored')

    def sfe_predict(self,fromnode,tonode,paths,model):
        """SFE预测"""
        self.Model = joblib.load(model)
        print('Model has been loaded')
        pre_data = {}
        i = 0
        for path in paths:
            pre_data['path'+str(i)] = self.sfe_compute_feature(path,fromnode,tonode)
            i += 1
        pre_data = pd.Series(pre_data).values.reshape(1,-1)
        print('pre_data:',pre_data,' shape:',pre_data.shape)
        print(self.Model.predict_proba(pre_data))
        print(self.Model.predict(pre_data))
        weights = self.Model.coef_
        print('Weights:',weights,'Bias:',self.Model.intercept_)

if __name__=="__main__":
    Military = Graph('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='military66')
 
    #def SFE_rw_model_test(Graph,predicted_relationship):
    #    SFE_rw = SFE_Model(Graph,predicted_relationship)
    #    #start_time = time.time()
    #    #paths = PRA.find_paths(mode='randomwalk',config={'length':3,'example_num':10,'sample_num':1000,'steps':20})
    #    #end_time =  time.time()
    #    #print("搜索用时:",end_time-start_time,'s')
    #    #with open('path_pra_country.txt','wb') as f:
    #    #    pickle.dump(paths,f)
    #    with open('path_pra_country.txt','rb') as f:
    #        paths = pickle.load(f)
    #    start_time = time.time()
    #    X_train,X_test,Y_train,Y_test = SFE_rw.sfe_data_construct(paths,60,2,{'data_name':'sfe_rw_data.txt'})
    #    end_time =  time.time()
    #    print("用时:",end_time-start_time,'s')
    #    start_time = time.time()
    #    #PRA.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':'sfe_rw_30.model'})
    #    SFE_rw.Model = joblib.load('sfe_rw_30.model')
    #    print('准确率:'+str(SFE_rw.Model.score(X_train+X_test,Y_train+Y_test)))
    #    end_time =  time.time()
    #    print("训练用时:",end_time-start_time,'s')

    #def SFE_BFS_model_test(Graph,predicted_relationship):
    #    SFE_BFS = SFE_Model(Graph,predicted_relationship)
    #    #start_time = time.time()
    #    #paths = SFE_BFS.find_paths(mode='BFS',config={'length':3,'example_num':10,'constraint':20})
    #    #end_time =  time.time()
    #    #print("搜索用时:",end_time-start_time,'s')
    #    #with open('path_sfe_bfs_country.txt','wb') as f:
    #    #    pickle.dump(paths,f)
    #    with open('path_sfe_bfs_country.txt','rb') as f:
    #        paths = pickle.load(f)
    #    start_time = time.time()
    #    X_train,X_test,Y_train,Y_test = SFE_BFS.sfe_data_construct(paths,30,157,{'data_name':'sfe_bfs_data.txt'})
    #    end_time =  time.time()
    #    print("用时:",end_time-start_time,'s')
    #    start_time = time.time()
    #    SFE_BFS.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':'sfe_bfs_30.model'})
    #    #SFE_BFS.Model = joblib.load('sfe_rw_30.model')
    #    print('准确率:'+str(SFE_BFS.Model.score(X_train+X_test,Y_train+Y_test)))
    #    end_time =  time.time()
    #    print("训练用时:",end_time-start_time,'s')    
    
    ##SFE_rw_model_test(Graph=Military,predicted_relationship=['生产单位位于','生产单位','产国'])
    #SFE_BFS_model_test(Graph=Military,predicted_relationship=['生产单位位于','生产单位','产国'])
   

    SFE = SFE_Model(Graph=Military,predicted_relationship=['生产单位位于','生产单位','产国'])
    ##paths = SFE.find_paths(mode='BFS',config={'length':3,'example_num':10,'constraint':20,'sample_num':1000,'strategy':'stratified'})
    
    #with open('path.txt','wb') as f:
    #    pickle.dump(paths,f)

    with open('path.txt','rb') as f:
        paths = pickle.load(f)

    SFE.Model = joblib.load('sfe_60_1.model')
    time_start = time.time() #开始计时
    X_train,X_test,Y_train,Y_test = SFE.sfe_data_construct(paths,60,38,{'data_name':'dataset_newlabel_60.txt'})
    time_end = time.time() #开始计时
    print('构建训练集总用时:',time_end-time_start,'s')
    print('准确率:'+str(SFE.Model.score(X_test,Y_test)))
    #weights = SFE.Model.coef_
    #y=weights.argsort()
    #for sample in y[0][:5]:
    #    print('Top weight',weights[0][sample],'path',paths[sample])
    #for sample in y[0][-5:]:
    #    print('Bottom weight',weights[0][sample],'path',paths[sample])
    #print('Weights:',weights,'Bias:',self.Model.intercept_)

    #SFE.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':'sfe_60_1.model'})
    #SFE.sfe_predict(15863,34,paths,'sfe_60.model')

    #SFE.sfe_compute_feature(({'relationship':'r生产单位'},{'relationship':'产国'}),3533,17)


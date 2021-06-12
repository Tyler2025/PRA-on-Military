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

    def sfe_data_construct(self,paths,scale,config,mode):
        """
        训练集与测试集构建
        paths:路径
        scale:抽样数量
        config:配置杂项
        """
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
                    path_feature = self.sfe_compute_feature(path,node_pair[0],node_pair[1])#计算路径特征值
                    #if i == main_col:
                    #    if path_feature == 0:#判断关系标签,只运行一遍
                    #        labels = labels + [0]
                    #    else:
                    #        labels = labels + [1]
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
    
    def sfe_acc_path_select(self,paths,label_file,path_num):
        """
        通过训练集标签中的数据进一步筛选路径
        paths:待选择的路径
        label_file:正例集合
        path_num:返回路径个数
        """
        potential_paths = []
        paths_acc = [0]*len(paths)
        path_acc_temp = []
        labels_qualified = self.load_labels('Labels.csv',label_file,1)#首先读出标签集
        n = len(labels_qualified)
        if path_num < len(paths):
            for i,path in enumerate(paths):
                for label in labels_qualified:
                    path_acc_temp.append(self.sfe_compute_feature(path,label[0],label[1]))
                paths_acc[i] = sum(path_acc_temp)/n
                path_acc_temp = []
            index = sorted(range(len(paths_acc)), key=lambda k: paths_acc[k],reverse = True)#返回降序排列acc的索引
            for index_num in index[:path_num]:
                potential_paths.append(paths[index_num])
            return potential_paths
        else:
            return paths

    def sfe_train(self,X_train,Y_train,X_test,Y_test,config):
        """SFE训练"""
        print('X_train:',X_train.shape)
        print('Y_train:',Y_train.shape)
        print('X_test:',X_test.shape)
        print('Y_test:',Y_test.shape)
        model = LogisticRegression(verbose=1)#实例化逻辑回归分类器
        model.fit(X_train,Y_train)
        self.Model = model
        precision = str(model.score(X_test,Y_test))
        print('准确率:'+precision)
        joblib.dump(model,config['model_name'])#保存模型为sfe.model
        print('Model has been stored')
        return precision

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
        proba = self.Model.predict_proba(pre_data)
        print(proba)
        predict = self.Model.predict(pre_data)
        print(predict)
        return pre_data,proba,predict
        #weights = self.Model.coef_
        #print('Weights:',weights,'Bias:',self.Model.intercept_)

def SFE_rw_model_test(Graph,predicted_relationship,mode,file_path,predict_nodes_pair=None,config=None):
    """
    mode:0-训练，1-预测
    """
    SFE_rw = SFE_Model(Graph,predicted_relationship)
    if mode == 0:
        start_time = time.time()
        paths = SFE_rw.find_paths(mode='randomwalk',config={'length':3,'example_num':10,'sample_num':1000,'steps':20})
        end_time =  time.time()
        search_time = end_time-start_time
        print("搜索用时:",search_time,'s')
        with open(file_path+'/path/path_sfe_randomwalk.txt','wb') as f:
            pickle.dump(paths,f)
        start_time = time.time()
        X_train,X_test,Y_train,Y_test = SFE_rw.sfe_data_construct(paths,config['num'],{'data_name':file_path+'/dataset/sfe_rw_data.txt'},mode=0)
        end_time =  time.time()
        construct_time = end_time-start_time
        print("构建训练集用时:",construct_time,'s')
        start_time = time.time()
        precision = SFE_rw.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':file_path+'/model/sfe_rw_45.model'})
        end_time =  time.time()
        print("训练用时:",end_time-start_time,'s')
        data_shape = X_train.shape +X_test.shape +Y_train.shape +Y_test.shape
        return paths,search_time,construct_time,precision,data_shape
    elif mode == 1:
        with open(file_path+'/path/path_sfe_randomwalk.txt','rb') as f:
            paths = pickle.load(f)
        start_time = time.time()
        pre_data,proba,predict= SFE_rw.sfe_predict(predict_nodes_pair[0],predict_nodes_pair[1],paths,file_path+'/model/sfe_rw_45.model')
        end_time =  time.time()
        print("预测用时:",end_time-start_time,'s')
        return pre_data,proba,predict

def PRA_model_test(Graph,predicted_relationship,mode,file_path,predict_nodes_pair=None,config=None):
    """
    mode:0-训练，1-预测
    """
    PRA = SFE_Model(Graph,predicted_relationship)
    if mode == 0:
        start_time = time.time()
        paths = PRA.find_paths(mode='randomwalk',config={'length':3,'example_num':10,'sample_num':1000,'steps':20})
        end_time =  time.time()
        search_time = end_time-start_time
        print("搜索用时:",search_time,'s')
        with open(file_path+'/path/path_pra_randomwalk.txt','wb') as f:
            pickle.dump(paths,f)
        #with open(file_path+'/path/path_pra_randomwalk.txt','rb') as f:
        #    paths = pickle.load(f)
        start_time = time.time()
        X_train,X_test,Y_train,Y_test = PRA.data_construt(paths,config['num'],{'data_name':file_path+'/dataset/pra_rw_data.txt'},mode=0)
        end_time =  time.time()
        construct_time = end_time-start_time
        print("构建训练集用时:",construct_time,'s')
        start_time = time.time()
        precision = PRA.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':file_path+'/model/pra_rw_45.model'})
        end_time =  time.time()
        print("训练用时:",end_time-start_time,'s')
        data_shape = X_train.shape +X_test.shape +Y_train.shape +Y_test.shape
        return paths,search_time,construct_time,precision,data_shape
    elif mode == 1:
        with open(file_path+'/path/path_pra_randomwalk.txt','rb') as f:
            paths = pickle.load(f)
        start_time = time.time()
        pre_data,proba,predict = PRA.predict(predict_nodes_pair[0],predict_nodes_pair[1],paths,file_path+'/model/pra_rw_45.model')
        end_time =  time.time()
        print("预测用时:",end_time-start_time,'s')
        return pre_data,proba,predict

def SFE_BFS_model_test(Graph,predicted_relationship,mode,file_path,predict_nodes_pair=None,config=None):
    """
    mode:0-训练，1-预测
    """
    SFE_BFS = SFE_Model(Graph,predicted_relationship)
    if mode == 0:
        start_time = time.time()
        paths = SFE_BFS.find_paths(mode='BFS',config={'length':3,'example_num':10,'constraint':20})
        end_time =  time.time()
        search_time = end_time-start_time
        print("搜索用时:",search_time,'s')
        with open(file_path+'/path/path_sfe_bfs_country.txt','wb') as f:
            pickle.dump(paths,f)
        #with open(file_path+'/path/path_sfe_bfs_country.txt','rb') as f:
        #    paths = pickle.load(f)
        start_time = time.time()
        X_train,X_test,Y_train,Y_test = SFE_BFS.sfe_data_construct(paths,config['num'],{'data_name':file_path+'/dataset/sfe_bfs_data.txt'},mode=0)
        end_time =  time.time()
        construct_time = end_time-start_time
        print("构建训练集用时:",construct_time,'s')
        start_time = time.time()
        precision = SFE_BFS.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':file_path+'/model/sfe_bfs_45.model'})
        end_time =  time.time()
        print("训练用时:",end_time-start_time,'s')    
        data_shape = X_train.shape +X_test.shape +Y_train.shape +Y_test.shape
        return paths,search_time,construct_time,precision,data_shape
    elif mode == 1:
        with open(file_path+'/path/path_sfe_bfs_country.txt','rb') as f:
            paths = pickle.load(f)
        start_time = time.time()
        pre_data,proba,predict = SFE_BFS.sfe_predict(predict_nodes_pair[0],predict_nodes_pair[1],paths,file_path+'/model/sfe_bfs_45.model')
        end_time =  time.time()
        print("预测用时:",end_time-start_time,'s')
        return  pre_data,proba,predict

def Military_inference(Graph,predicted_relationship,mode,file_path,predict_nodes_pair=None,config=None):
    """
    mode:0-训练，1-预测
    """
    SFE = SFE_Model(Graph,predicted_relationship)
    if mode == 0:
        time_start = time.time() #开始计时
        paths = SFE.find_paths(mode='BFS',config={'length':3,'example_num':10,'constraint':20,'sample_num':1000,'strategy':'stratified'})
        paths = SFE.sfe_acc_path_select(paths,'labels.txt',10)
        time_end = time.time() 
        search_time = time_end-time_start
        print('路径选择用时:',search_time,'s')
        with open(file_path+'/path/sfe_path_acc_select30_deletenull.txt','wb') as f:
            pickle.dump(paths,f)
        time_start = time.time() #开始计时
        X_train,X_test,Y_train,Y_test = SFE.sfe_data_construct(paths,config['num'],{'data_name':'dataset_path30_newlabel_45_1_deletenull.txt'},mode=0)
        time_end = time.time() 
        construct_time = time_end-time_start
        print('构建训练集总用时:',construct_time,'s')
        start_time = time.time()
        precision = SFE.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':file_path+'/model/MRRA_path30_newlabel_45_1_deletenull.model'})
        end_time =  time.time()
        print("训练用时:",end_time-start_time,'s')    
        data_shape = X_train.shape +X_test.shape +Y_train.shape +Y_test.shape
        return paths,search_time,construct_time,precision,data_shape
        #weights = SFE.Model.coef_
        #y = weights.argsort()
        #for sample in y[0][:5]:
        #    print('Top weight',weights[0][sample],'path',paths[sample])
        #for sample in y[0][-5:]:
        #    print('Bottom weight',weights[0][sample],'path',paths[sample])
    elif mode == 1:
        with open(file_path+'/path/sfe_path_acc_select30_deletenull.txt','rb') as f:
            paths = pickle.load(f)
        start_time = time.time()
        pre_data,proba,predict = SFE.sfe_predict(predict_nodes_pair[0],predict_nodes_pair[1],paths,file_path+'/model/dataset_path30_newlabel_45_1_deletenull.model')
        end_time =  time.time()
        print("预测用时:",end_time-start_time,'s')
        return  pre_data,proba,predict

if __name__=="__main__":
    Military = Graph('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='military66')
 
    #def PRA_model_test(Graph,predicted_relationship,mode,file_path,predict_nodes_pair=None):
    #    """
    #    mode:0-训练，1-预测
    #    """
    #    PRA = SFE_Model(Graph,predicted_relationship)
    #    if mode == 0:
    #        #start_time = time.time()
    #        #paths = PRA.find_paths(mode='randomwalk',config={'length':3,'example_num':10,'sample_num':1000,'steps':20})
    #        #end_time =  time.time()
    #        #print("搜索用时:",end_time-start_time,'s')
    #        #with open(file_path+'/path/path_pra_randomwalk.txt','wb') as f:
    #        #    pickle.dump(paths,f)
    #        with open(file_path+'/path/path_pra_randomwalk.txt','rb') as f:
    #            paths = pickle.load(f)
    #        start_time = time.time()
    #        X_train,X_test,Y_train,Y_test = PRA.data_construt(paths,45,{'data_name':file_path+'/dataset/pra_rw_data.txt'},mode=0)
    #        end_time =  time.time()
    #        print("构建训练集用时:",end_time-start_time,'s')
    #        start_time = time.time()
    #        PRA.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':file_path+'/model/pra_rw_45.model'})
    #        end_time =  time.time()
    #        print("训练用时:",end_time-start_time,'s')
    #    elif mode == 1:
    #        with open(file_path+'/path/path_pra_randomwalk.txt','rb') as f:
    #            paths = pickle.load(f)
    #        start_time = time.time()
    #        SFE_rw.sfe_predict(predict_nodes_pair[0],predict_nodes_pair[1],paths,file_path+'/model/pra_rw_45.model')
    #        end_time =  time.time()
    #        print("预测用时:",end_time-start_time,'s')
    #        return

    #def SFE_rw_model_test(Graph,predicted_relationship,mode,file_path,predict_nodes_pair=None):
    #    """
    #    mode:0-训练，1-预测
    #    """
    #    SFE_rw = SFE_Model(Graph,predicted_relationship)
    #    if mode == 0:
    #        start_time = time.time()
    #        paths = SFE_rw.find_paths(mode='randomwalk',config={'length':3,'example_num':10,'sample_num':1000,'steps':20})
    #        end_time =  time.time()
    #        print("搜索用时:",end_time-start_time,'s')
    #        with open(file_path+'/path/path_sfe_randomwalk.txt','wb') as f:
    #            pickle.dump(paths,f)
    #        start_time = time.time()
    #        X_train,X_test,Y_train,Y_test = SFE_rw.sfe_data_construct(paths,45,2,{'data_name':file_path+'/dataset/sfe_rw_data.txt'},mode=0)
    #        end_time =  time.time()
    #        print("构建训练集用时:",end_time-start_time,'s')
    #        start_time = time.time()
    #        SFE_rw.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':file_path+'/model/sfe_rw_45.model'})
    #        end_time =  time.time()
    #        print("训练用时:",end_time-start_time,'s')
    #    elif mode == 1:
    #        with open(file_path+'/path/path_sfe_randomwalk.txt','rb') as f:
    #            paths = pickle.load(f)
    #        start_time = time.time()
    #        pre_data,proba,predict= SFE_rw.sfe_predict(predict_nodes_pair[0],predict_nodes_pair[1],paths,file_path+'/model/sfe_rw_45.model')
    #        end_time =  time.time()
    #        print("预测用时:",end_time-start_time,'s')
    #        return pre_data,proba,predict

    #def SFE_BFS_model_test(Graph,predicted_relationship,mode,file_path,predict_nodes_pair=None):
    #    """
    #    mode:0-训练，1-预测
    #    """
    #    SFE_BFS = SFE_Model(Graph,predicted_relationship)
    #    if mode == 0:
    #        #start_time = time.time()
    #        #paths = SFE_BFS.find_paths(mode='BFS',config={'length':3,'example_num':10,'constraint':20})
    #        #end_time =  time.time()
    #        #print("搜索用时:",end_time-start_time,'s')
    #        #with open('path_sfe_bfs_country.txt','wb') as f:
    #        #    pickle.dump(paths,f)
    #        with open(file_path+'/path/path_sfe_bfs_country.txt','rb') as f:
    #            paths = pickle.load(f)
    #        start_time = time.time()
    #        X_train,X_test,Y_train,Y_test = SFE_BFS.sfe_data_construct(paths,45,2,{'data_name':file_path+'/dataset/sfe_bfs_data.txt'},mode=0)
    #        end_time =  time.time()
    #        print("构建训练集用时:",end_time-start_time,'s')
    #        start_time = time.time()
    #        SFE_BFS.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':file_path+'/model/sfe_bfs_45.model'})
    #        print('准确率:'+str(SFE_BFS.Model.score(X_test,Y_test)))
    #        end_time =  time.time()
    #        print("训练用时:",end_time-start_time,'s')    
    #    elif mode == 1:
    #        with open(file_path+'/path/path_sfe_bfs_country.txt','rb') as f:
    #            paths = pickle.load(f)
    #        start_time = time.time()
    #        SFE_BFS.sfe_predict(predict_nodes_pair[0],predict_nodes_pair[1],paths,file_path+'/model/sfe_bfs_45.model')
    #        end_time =  time.time()
    #        print("预测用时:",end_time-start_time,'s')
    #        return  

    #def Military_inference(Graph,predicted_relationship,mode):
    #    """
    #    mode:0-训练，1-预测
    #    """
    #    SFE = SFE_Model(Graph=Military,predicted_relationship=['生产单位位于','生产单位','产国'])
    #    with open('sfe_path_acc_select30_deletenull.txt','rb') as f:
    #        paths = pickle.load(f)
    #    time_start = time.time() #开始计时
    #    useful_paths = SFE.sfe_acc_path_select(paths,'labels.txt',30)
    #    time_end = time.time() 
    #    print('acc筛选用时:',time_end-time_start,'s')
    #    time_start = time.time() #开始计时
    #    X_train,X_test,Y_train,Y_test = SFE.sfe_data_construct(paths,45,38,{'data_name':'dataset_path30_newlabel_45_1_deletenull.txt'})
    #    time_end = time.time() 
    #    print('构建训练集总用时:',time_end-time_start,'s')
    #    weights = SFE.Model.coef_
    #    y = weights.argsort()
    #    for sample in y[0][:5]:
    #        print('Top weight',weights[0][sample],'path',paths[sample])
    #    for sample in y[0][-5:]:
    #        print('Bottom weight',weights[0][sample],'path',paths[sample])
    SFE_rw_model_test(Graph=Military,predicted_relationship=['生产单位位于','生产单位','产国'],mode=1,file_path='Temp_Files/SFE_rw',predict_nodes_pair=[3533,17])
    #PRA_model_test(Graph=Military,predicted_relationship=['生产单位位于','生产单位','产国'],mode=0,file_path='Temp_Files/PRA')
    #SFE_BFS_model_test(Graph=Military,predicted_relationship=['生产单位位于','生产单位','产国'],mode=0,file_path='Temp_Files/SFE_bfs')
   

    #SFE = SFE_Model(Graph=Military,predicted_relationship=['生产单位位于','生产单位','产国'])
    ##paths = SFE.find_paths(mode='BFS',config={'length':3,'example_num':10,'constraint':20,'sample_num':1000,'strategy':'stratified'})
    
    #with open('path.txt','wb') as f:
    #    pickle.dump(paths,f)

    #with open('sfe_path_acc_select30_deletenull.txt','rb') as f:
    #    paths = pickle.load(f)

    #time_start = time.time() #开始计时
    #useful_paths = SFE.sfe_acc_path_select(paths,'labels.txt',30)
    #time_end = time.time() 
    #print('acc筛选用时:',time_end-time_start,'s')
    ##with open('sfe_path_acc_select30_deletenull.txt','wb') as f:
    ##    pickle.dump(useful_paths,f)
    #print(useful_paths)
    #time_start = time.time() #开始计时
    #X_train,X_test,Y_train,Y_test = SFE.sfe_data_construct(paths,45,38,{'data_name':'dataset_path30_newlabel_45_1_deletenull.txt'})
    #time_end = time.time() 
    #print('构建训练集总用时:',time_end-time_start,'s')
    #weights = SFE.Model.coef_
    #y=weights.argsort()
    #for sample in y[0][:5]:
    #    print('Top weight',weights[0][sample],'path',paths[sample])
    #for sample in y[0][-5:]:
    #    print('Bottom weight',weights[0][sample],'path',paths[sample])
    #print('Weights:',weights,'Bias:',self.Model.intercept_)

    #SFE.sfe_train(X_train,Y_train,X_test,Y_test,{'model_name':'dataset_path30_newlabel_45_1_deletenull.model'})
    #SFE.Model = joblib.load('sfe_newlabel_45_1.model')
    #print('Model has been loaded')
    #weights = SFE.Model.coef_
    #y=weights.argsort()
    #for sample in y[0][:5]:
    #    print('Top weight',weights[0][sample],'path',paths[sample])
    #for sample in y[0][-5:]:
    #    print('Bottom weight',weights[0][sample],'path',paths[sample])

    #SFE.sfe_compute_feature(({'relationship':'r生产单位'},{'relationship':'产国'}),3533,17)


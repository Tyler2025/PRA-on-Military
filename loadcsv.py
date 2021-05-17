
# coding=utf-8
from py2neo import Graph
import os
import csv

#创建图数据库实例，并建立链接
#military = Graph("http://localhost:7474",username="neo4j",password="ne1013pk250")

def get_csvfiles():
    csv_files=[]
    files = os.listdir()
    for file in files:
        if file[-4:] == '.csv':
            csv_files.append(file)
    return csv_files

class Importoneo4j():
    def __init__(self,url,auth,name):
        self.url = url
        self.auth = auth
        self.name = name
        self.graph = Graph(url,auth=auth,name=name)

    def import_data(self):
        load_str = 'LOAD CSV WITH HEADERS FROM '
        file_list = get_csvfiles()
        for file in file_list:
            print(file)
            with open(file,encoding='utf-8',newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    break
                print(row)
                row.remove('_id')
                print('row=',row)
            target = load_str+"'file:///csv/"+file+"' "+'AS line '
            label = file[:-4].replace('（','')
            label = label.replace('）','')
            target2 = 'MERGE (:'+label+'{'+str([pro+':line.'+pro for pro in row])+'})'
            target2 = target2.replace("'",'')
            target2 = target2.replace("[",'')
            target2 = target2.replace("]",'')
            print(target,'\n'+target2)
            #self.graph.run(target+target2)
            #self.graph.run(target2)
            print(file+'Import Done')

    def import_data_mi(self):
        file_list = get_csvfiles()
        loc = 1
        for file in file_list:
            flag = 0 #解析flag
            name_index = 0
            id_index = 0
            property = []
            print('Import'+file+str(loc)+'/'+str(len(file_list)))
            with open(file,encoding='utf-8',newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    if flag == 0:#首先将表头暂存起来，并查找特殊列的索引
                        property = row
                        flag += 1
                        name_index = property.index('名称')
                        id_index = property.index('_id')
                        index_l = list(range(len(property)-2))
                        index_l.remove(id_index)
                        index_l.remove(name_index)
                    else:
                        for i in index_l:
                            label = file[:-4].replace('（','').replace(')','')
                            query = "MERGE (m:`"+label+"`{名称:\""+row[name_index].replace('\"','”')+"\"}) MERGE (n:`"+property[i]+"`{名称:\""+row[i].replace('\"','”')+"\"}) MERGE (m)-[r:`"+property[i]+"`]->(n) MERGE (n)-[x:`r"+property[i]+"`]->(m)"
                            self.graph.run(query)
            loc += 1
military_graph = Importoneo4j('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='military66')
military_graph.import_data_mi()
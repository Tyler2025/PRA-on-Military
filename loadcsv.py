
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
            with open(file,encoding='utf-8',newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    break
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
military_graph = Importoneo4j('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='Military')
military_graph.import_data()
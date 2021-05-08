# coding=utf-8
'''
为军事图谱创建关系
'''
from py2neo import Graph
import os
import csv
from translate import Translator

class Createoneo4j():
    def __init__(self,url,auth,name):
        self.url = url
        self.auth = auth
        self.name = name
        self.graph = Graph(url,auth=auth,name=name)

    def import_data(self,filename):
        load_str = 'LOAD CSV WITH HEADERS FROM '
        file_list = filename
        for file in file_list:
            with open(file,newline='',encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    break
                #row.remove('_id')
                #print('row=',row)
            target = load_str+"'file:///csv/"+file+"' "+'AS line '
            label = file[:-4].replace('（','')
            label = label.replace('）','')
            target2 = 'MERGE (:'+label+'{'+str([pro+':line.'+pro for pro in row])+'})'
            target2 = target2.replace("'",'')
            target2 = target2.replace("[",'')
            target2 = target2.replace("]",'')
            print(target,'\n'+target2)
            self.graph.run(target+target2)
            #self.graph.run(target2)
            #print(file+'Import Done')

    def create_relationships(self):
            target2 = 'MATCH (n),(累心:Categories) WHERE 累心.种类 = 砸蛋.类型 MERGE (砸蛋)-[r:类型是]->(累心)'+')'
            print(target2)
            #self.graph.run(target+target2)
            #self.graph.run(target2)
            #print(file+'Import Done')   
                 
military_graph = Createoneo4j('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='Military')
#cursor = military_graph.graph.call.db.labels()
military_graph.import_data(['生产单位.csv'])

'''
translator= Translator(from_lang="chinese",to_lang="english")
print(translator.translate('法国'))
'''
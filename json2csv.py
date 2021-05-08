# coding=utf-8

import pandas as pd
import json
import csv
filename = 'military.json'

#定义属性名
properties={ }

#读入json文件,json_data为包含5800个字典的列表
with open(filename,'r',encoding="utf-8") as f_ojb:
    json_data = json.load(f_ojb)
    #对列表中的对象遍历操作,获取类型
    for weapon in json_data:
        types = weapon["类型"]
        if types in properties.keys():
            properties[types] = list(set(properties[types]+list(weapon.keys()))) 
        else:
            properties[types] = list(weapon.keys())

#print(properties)
#print(len(properties))

#开始写入CSV文件
for types in properties.keys():
    with open(types.replace('/','')+'.csv','w',encoding='utf-8',newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(properties[types])
        for weapon in json_data:
            if weapon["类型"] == types:
                t = []
                for key in properties[types]:
                    if (key not in list(weapon.keys())) or (weapon[key] == ''):
                        t.append("NULL")
                    else:                     
                        t.append(weapon[key])
                csv_writer.writerow(t)

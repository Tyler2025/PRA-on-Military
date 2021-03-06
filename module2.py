# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'module2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import sys
import os
from PyQt5.QtWidgets import QApplication,QMainWindow,QHBoxLayout,QFileDialog
from PyQt5.QtWebEngineWidgets  import *
from PyQt5.QtCore import QUrl
from SFE_Train_Form import Ui_MainWindow
from SFE_Train import SFE_rw_model_test,PRA_model_test,SFE_BFS_model_test,Military_inference
from py2neo import Graph 

class Example(Ui_MainWindow):
    def __init__(self):
        super(Example,self).__init__()
        self.cwd = os.getcwd()
        self.graph = Graph('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='military66')

    def init_ui(self):
        self.browser1.load(QUrl('http://localhost:7474'))
        self.pushButton_4.clicked.connect(self.get_train_dir)
        self.pushButton.clicked.connect(self.get_predict_dir)
        self.pushButton_3.clicked.connect(self.start_predict)
        self.pushButton_2.clicked.connect(self.start_train)

    def get_train_dir(self):
        dir_choose = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", self.cwd)
        self.lineEdit_3.setText(dir_choose)

    def get_predict_dir(self):
        dir_choose = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", self.cwd)
        self.lineEdit_4.setText(dir_choose)

    def start_predict(self):
        pre_data_string = ""
        predict_string=""
        path = self.lineEdit_4.text()#路径
        source_node = self.lineEdit.text()
        target_node = self.lineEdit_2.text()
        if len(source_node) == 0 or len(path) == 0 or len(target_node)==0:
            print("请选择路径并输入预测节点对ID")
            self.statusbar.showMessage('请选择路径并输入预测节点对ID', 2000)
        else:
            predict_nodes_pair = [int(source_node),int(target_node)]#节点对
            mode = 1
            index = self.comboBox_4.currentIndex()
            if index == 0:#PRA
                pre_data,proba,predict = PRA_model_test(self.graph,['生产单位位于','生产单位','产国'],1,path,predict_nodes_pair)
                self.textBrowser_4.setText('不成立:'+str(proba[0][0])+'\r\n'+'成立:'+str(proba[0][1]))
                if predict[0] == 0:
                    predict_string = '不成立'
                elif predict[0] == 1:
                    predict_string = '成立'
                self.textBrowser_5.setText(predict_string)
                for i,predatas in enumerate(pre_data[0]):
                    pre_data_string =pre_data_string + "path"+str(i)+": "+str(predatas)+"\r\n"
                self.textBrowser_3.setText(pre_data_string)
            elif index == 1:#SFE_rw
                pre_data,proba,predict = SFE_rw_model_test(self.graph,['生产单位位于','生产单位','产国'],1,path,predict_nodes_pair)
                self.textBrowser_4.setText('不成立:'+str(proba[0][0])+'\r\n'+'成立:'+str(proba[0][1]))
                if predict[0] == 0:
                    predict_string = '不成立'
                elif predict[0] == 1:
                    predict_string = '成立'
                self.textBrowser_5.setText(predict_string)
                for i,predatas in enumerate(pre_data[0]):
                    pre_data_string =pre_data_string + "path"+str(i)+": "+str(predatas)+"\r\n"
                self.textBrowser_3.setText(pre_data_string)
            elif index == 2:#SFE_BFS
                pre_data,proba,predict = SFE_BFS_model_test(self.graph,['生产单位位于','生产单位','产国'],1,path,predict_nodes_pair)
                self.textBrowser_4.setText('不成立:'+str(proba[0][0])+'\r\n'+'成立:'+str(proba[0][1]))
                if predict[0] == 0:
                    predict_string = '不成立'
                elif predict[0] == 1:
                    predict_string = '成立'
                self.textBrowser_5.setText(predict_string)
                for i,predatas in enumerate(pre_data[0]):
                    pre_data_string =pre_data_string + "path"+str(i)+": "+str(predatas)+"\r\n"
                self.textBrowser_3.setText(pre_data_string)
            elif index == 3:#Mili
                pre_data,proba,predict = Military_inference(self.graph,['生产单位位于','生产单位','产国'],1,path,predict_nodes_pair)
                self.textBrowser_4.setText('不成立:'+str(proba[0][0])+'\r\n'+'成立:'+str(proba[0][1]))
                if predict[0] == 0:
                    predict_string = '不成立'
                elif predict[0] == 1:
                    predict_string = '成立'
                self.textBrowser_5.setText(predict_string)
                for i,predatas in enumerate(pre_data[0]):
                    pre_data_string =pre_data_string + "path"+str(i)+": "+str(predatas)+"\r\n"
                self.textBrowser_3.setText(pre_data_string)

    def start_train(self):
        path = self.lineEdit_3.text()#路径
        config = {}
        path_string = ""
        result_string = ""
        if len(path) == 0:
            print("请选择路径")
            self.statusbar.showMessage('请选择路径', 2000)
        else:
            self.mkdir(path+'/path/')
            self.mkdir(path+'/dataset/')
            self.mkdir(path+'/model/')
            index = self.comboBox_2.currentIndex()
            config['num'] = self.spinBox.value()
            if index == 0:#PRA
                paths,search_time,construct_time,precision,data_shape = PRA_model_test(self.graph,['生产单位位于','生产单位','产国'],0,path,config=config)
                for i,path in enumerate(paths):
                    path_string =path_string + "path"+str(i)+": "+str(path)+"\r\n"
                self.textBrowser_6.setText(path_string)#显示找到的路径
                result_string = "搜索时间："+str(search_time)+"\r\n"+"数据集构建时间："+str(construct_time)+"\r\n"+"准确率："+precision+"\r\n"+"训练集形状："+str(data_shape[0:2])+"\r\n"+"测试集形状："+str(data_shape[2:4])
                self.textBrowser_2.setText(result_string)
            elif index == 1:#SFE_RW
                paths,search_time,construct_time,precision,data_shape = SFE_rw_model_test(self.graph,['生产单位位于','生产单位','产国'],0,path,config=config)
                for i,path in enumerate(paths):
                    path_string =path_string + "path"+str(i)+": "+str(path)+"\r\n"
                self.textBrowser_6.setText(path_string)#显示找到的路径
                result_string = "搜索时间："+str(search_time)+"\r\n"+"数据集构建时间："+str(construct_time)+"\r\n"+"准确率："+precision+"\r\n"+"训练集形状："+str(data_shape[0:2])+"\r\n"+"测试集形状："+str(data_shape[2:4])
                self.textBrowser_2.setText(result_string)
            elif index ==2:#SFE_BFS
                paths,search_time,construct_time,precision,data_shape = SFE_BFS_model_test(self.graph,['生产单位位于','生产单位','产国'],0,path,config=config)
                for i,path in enumerate(paths):
                    path_string =path_string + "path"+str(i)+": "+str(path)+"\r\n"
                self.textBrowser_6.setText(path_string)#显示找到的路径
                result_string = "搜索时间："+str(search_time)+"\r\n"+"数据集构建时间："+str(construct_time)+"\r\n"+"准确率："+precision+"\r\n"+"训练集形状："+str(data_shape[0:2])+"\r\n"+"测试集形状："+str(data_shape[2:4])
                self.textBrowser_2.setText(result_string)
            elif index == 3:#MRRA
                paths,search_time,construct_time,precision,data_shape = Military_inference(self.graph,['生产单位位于','生产单位','产国'],0,path,config=config)
                for i,path in enumerate(paths):
                    path_string =path_string + "path"+str(i)+": "+str(path)+"\r\n"
                self.textBrowser_6.setText(path_string)#显示找到的路径
                result_string = "搜索时间："+str(search_time)+"\r\n"+"数据集构建时间："+str(construct_time)+"\r\n"+"准确率："+precision+"\r\n"+"训练集形状："+str(data_shape[0:2])+"\r\n"+"测试集形状："+str(data_shape[2:4])
                self.textBrowser_2.setText(result_string)
                pass

    def mkdir(self,path):
        folder = os.path.exists(path)
        if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
            print("New Folder Been Created!!")
        else:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Example()
    ui.setupUi(mainWindow)
    ui.init_ui()
    mainWindow.show()
    sys.exit(app.exec_())


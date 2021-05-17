# coding=utf-8
'''
为SFE算法训练逻辑回归模型
'''
Military = Graph('http://localhost:7474',auth=('neo4j','ne1013pk250'),name='neo4j')

class SFE_Model():
    """
    SFE算法模型类
    """
    def __init__(self, *args, **kwargs):
        """
        初始化SFE类，
        """
        return super().__init__(*args, **kwargs)
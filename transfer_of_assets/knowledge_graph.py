import pandas as pd
import numpy as np
from py2neo import Graph,Node,Relationship,NodeMatcher,Subgraph
from py2neo.matching import RelationshipMatcher
import openpyxl

file1 = pd.read_excel("交付使用资产清册确认表-模板.xls",sheet_name='平台地面集输工程模板',header=2)
file2 = pd.read_excel("sheet3.xlsx",sheet_name='Sheet1')

# 修改属性名
file1.rename(columns={'名称':'资产名称'},inplace = True)

# 删除不需要的行
for i in range(0,len(file1)):
    cell = file1.loc[i,'数量']
    f_cell = float(cell)
    if np.isnan(f_cell) == True:
        file1.drop(index=i, axis=0, inplace=True)

# 把模板中的资产名称抽取出来
asset_name_list = []
for i in range(1, len(file1)):
    asset_name_list.append(file1['资产名称'][i])
asset_name_list = list(set(asset_name_list)) #去重

# 删除表3中的所有属性内容
for i in range(0,len(file2)):
    file2.drop(index=i, axis=0, inplace=True)

# 删除表3不需要的属性
file2.drop(['资产名称'],axis=1, inplace=True)
file2.drop(['合同编号（待用项2）'],axis=1, inplace=True)
file2.drop(['合同名称（待用项3）'],axis=1, inplace=True)

# 抽取表3中需要的属性名称存入asset_property_list中
asset_property_list = []
for i in range(0,66):
    asset_property_list.append(file2.columns[i])

# 先设置所有属性内容为空
asset_value_list = []
for i in range(0, len(file2)):
    for j in range(0, 66):
        # file2[file2.columns[j]][i] = '0'
        asset_value_list.append(file2[file2.columns[j]][i])
asset_value_list = list(set(asset_value_list))

# 转变为str类型
asset_value_list = [str(i) for i in asset_value_list]

print(asset_value_list)

#最关键的一步，把数据变成元组，按照(资产名称，资产属性，属性内容)进行排序
tuple_total = list(zip(asset_name_list,asset_property_list,asset_value_list))

#把数据导入Neo4j中
graph = Graph("http://localhost:7474", auth = ('neo4j','Shen767900'),name = 'neo4j')
# 清除neo4j里面的所有数据
graph.delete_all()

label_1 = '资产名称'
label_2 = '资产属性内容'

#把节点导入neo4j中
def create_node(asset_name_list, asset_value_list):
    for name in asset_name_list:
        node_1 = Node(label_1, name = name)
        graph.create(node_1)
    for name in asset_value_list:
        node_2 = Node(label_2, name = name)
        graph.create(node_2)
create_node(asset_name_list, asset_value_list)

# 导入关系
matcher = NodeMatcher(graph)
for i in range(0, len(tuple_total)):
    rel = Relationship(matcher.match(label_1, name = tuple_total[i][0]).first(),
                      tuple_total[i][1],
                      matcher.match(label_2, name = tuple_total[i][2]).first()
                      )
    graph.create(rel)

#创建完成
print('Finished!!!')

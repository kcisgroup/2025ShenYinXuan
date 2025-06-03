import pandas as pd
import numpy as np
from py2neo import Graph,Node,Relationship,NodeMatcher,Subgraph
from py2neo.matching import RelationshipMatcher
import openpyxl
from openpyxl import load_workbook
import data_processing
import re


# 连接neo4j数据库
graph = Graph("http://localhost:7474", auth = ('neo4j','*******'),name = 'neo4j') # 此处省略密码

# 加载Excel表
df = pd.read_excel("D:\intelligent transfer of capital\data of transfer of capital\sheet_merge.xlsx")

# 初始化输出表output和failed_match_assets、failed_match_properties
output = pd.DataFrame(columns=['资产名称'])
failed_match_assets = pd.DataFrame(columns=df.columns)
failed_match_properties = pd.DataFrame(columns=df.columns)
failed_match_properties_row = {}

# 遍历sheet_merge.xlsx的每一行
for index, row in df.iterrows():
    # 匹配资产名称
    asset_name = row['资产名称']
    query = f"MATCH (a: 资产名称 {{name: '{asset_name}'}}) RETURN a"
    result = graph.run(query)
    # 如果没有匹配成功，将该行复制到failed_match_assets表中
    if not result:
        failed_match_assets = failed_match_assets.append(row, ignore_index=True)
        # 删除sheet_merge中该行
        df.drop(index, inplace=True)
        continue
    else:
    # 如果匹配成功，输出到output表中
        output_row = {'资产名称': asset_name}

    # 遍历sheet_merge.xlsx的每个列名，进行资产属性匹配
    for col in df.columns:
        if col == '资产名称':
            continue
        attr_value = row[col]  #col是列名，row[col]是列名下面每行的内容
        query = f"MATCH (a:资产名称 {{name: '{asset_name}'}})-[r:`包含属性`]->(b:资产属性 {{name: '{col}'}}) RETURN b"
        result = graph.run(query)
        if result.data():
            output_row[col] = attr_value
        else:
            failed_match_properties_row[col] = attr_value
    # 将输出行添加到output表中
    output = output.append(output_row, ignore_index=True)
    # 将输出行添加到failed_match_properties表中
    failed_match_properties = failed_match_properties.append(failed_match_properties_row, ignore_index=True)

# 查找label="信息获取方式"、名为"固定值"的节点
query1 = '''
MATCH (n:信息获取方式 {name: "固定值"})-[:包含]->(m:资产属性)
RETURN m.name AS name
'''
result1 = graph.run(query1)

# 提取6个资产属性节点名称
column_names = [record['name'] for record in result1]

# 将6个资产属性节点名称作为列名插入到最后
output = pd.concat([output, pd.DataFrame(columns=column_names)], axis=1)

# 通过6个“资产属性”的“属性值”关系，查找到label=“固定值”的6个节点
query2 = '''
MATCH (n:资产属性)-[:属性值]->(m:固定值)
WHERE n.name IN $column_names
RETURN n.name AS name, m.name AS value
'''
result2 = graph.run(query2, column_names=column_names)

# 将查询结果转换为字典
data = {}
for record in result2:
    if record['name'] not in data:
        data[record['name']] = []
    data[record['name']].append(record['value'])

# 将查询结果作为output.xlsx的6列的列内容
for column in column_names:
    context ="".join(data.get(column)) #将list类型转换成字符串类型
    for i in range(len(output)):
        output.loc[i,column] = context

# 查找label="信息获取方式"、名为"自定义获取规则"的节点
query3 = '''
MATCH (n:信息获取方式 {name: "自定义获取规则"})-[:包含]->(m:资产属性)
RETURN m.name AS name
'''
result3 = graph.run(query3)

# 提取资产属性节点名称（原值、增加原因）
column_names_2 = [record['name'] for record in result3]

# 将资产属性节点名称作为列名插入到最后
output = pd.concat([output, pd.DataFrame(columns=column_names_2)], axis=1)

# 通过“资产属性”的“弃置资产”关系，查找到label=“增加原因”的节点
query4 = '''
MATCH (n:资产属性)-[:弃置资产]->(m:增加原因)
WHERE n.name IN $column_names
RETURN n.name AS name, m.name AS value
'''
result4 = graph.run(query4, column_names=column_names_2)

# 通过“资产属性”的“其他资产”关系，查找到label=“增加原因”的节点
query5 = '''
MATCH (n:资产属性)-[:其他资产]->(m:增加原因)
WHERE n.name IN $column_names
RETURN n.name AS name, m.name AS value
'''
result5 = graph.run(query5, column_names=column_names_2)

# 将查询结果转换为字典
data_2 = {}  #{'增加原因': ['22']}
for record in result4:
    if record['name'] not in data_2:
        data_2[record['name']] = []
    data_2[record['name']].append(record['value'])

data_3 = {}  #{'增加原因': ['20']}
for record in result5:
    if record['name'] not in data_3:
        data_3[record['name']] = []
    data_3[record['name']].append(record['value'])

for i in range(len(output)):
     # 提取前两个中文字，判断是否为“弃置”：是-22，否-20
    pattern = re.compile(u'[\u4e00-\u9fa5]{2}')
    cell_content = output.loc[i]['资产名称']
    match = pattern.search(cell_content)
    if match:
        first_two_chars = match.group()
        if first_two_chars=='弃置':
            for item in data_2.values():
                context1 = "".join(item)
                output.loc[i,'增加原因'] = context1
        else:
            for item in data_3.values():
                context2 = "".join(item)
                output.loc[i,'增加原因'] = context2

# 弃置资产 原值=参考原值
row_disposal = len(output)-1
output.loc[row_disposal,'原值'] = format(output.loc[row_disposal,'参考原值'],'.2f')

output.to_excel('D:/intelligent transfer of capital/data of transfer of capital/output.xlsx',index=False)
# 将failed_match表保存到failed_match.xlsx中
failed_match_assets.to_excel('D:/intelligent transfer of capital/data of transfer of capital/failed_match_assets.xlsx', index=False)
failed_match_properties.to_excel('D:/intelligent transfer of capital/data of transfer of capital/failed_match_properties.xlsx', index=False)

print('all finished!')


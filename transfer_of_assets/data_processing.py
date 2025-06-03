import pandas as pd
import numpy as np
import openpyxl
import xlrd

# file1 = 'sheet1.xlsx'
# file2 = 'sheet2.xlsx'
#
# # 跳过开头几行读取表1和表2
# shu_ju1_1 = pd.read_excel(file1,header=2) #header第几行开始读取（从0行开始算）
# shu_ju2_1 = pd.read_excel(file2,sheet_name='转表2',header=3)#sheet_name为表名称，即当Excel中有多个表时，选取其中一个表进行读取
# shu_ju2_9 = pd.read_excel(file2,sheet_name='转表9',header=13)

def Data_Processing(shu_ju1_1,shu_ju2_1,shu_ju2_9):

    # 删除表1和表2空白行
    shu_ju1_2 = shu_ju1_1.dropna(how="all")
    shu_ju2_2 = shu_ju2_1.dropna(how="all")

    # 表1表2修改属性名称
    shu_ju1_2.rename(columns={'名称':'资产名称'},inplace = True)
    shu_ju1_2.rename(columns={'资产所在地':'存放(安装地点)'},inplace = True)
    shu_ju1_2.rename(columns={'施工单位':'制造厂家'},inplace = True)
    shu_ju1_2.rename(columns={'数量':'复合数量'},inplace = True)
    shu_ju2_2.rename(columns={'规格及型号':'规格型号'},inplace = True)
    shu_ju2_2.rename(columns={'预转资金额':'参考原值'},inplace = True)

    # 在第一列的位置添加"资产编号"
    shu_ju1_2.insert(1, '资产编号','nan')
    shu_ju2_2.insert(1, '资产编号','nan')

    # 表1自动生成“资产编号”
    row_index = 1
    cnt = 0 # 已编号的行索引
    for i in range(1,len(shu_ju1_2)):
        if i > cnt-1:
            j = 1
            shu_ju1_2.loc[i, "资产编号"] = str(row_index) + '_' + str(j)
            for x in range(i+1,len(shu_ju1_2)):
                if shu_ju1_2.loc[i,"资产名称"] == shu_ju1_2.loc[x,"资产名称"]:
                    j += 1
                    shu_ju1_2.loc[x,"资产编号"] = str(row_index) + '_' + str(j)
                else:
                    cnt = x
                    break
            row_index += 1

    # 表2根据“资产名称”匹配表1的资产编号
    for i in range(1,len(shu_ju2_2)):
        if i!=1 and shu_ju2_2.loc[i, '资产名称'] != shu_ju2_2.loc[i - 1, '资产名称'] :
            count = 0
        for m in range(1,len(shu_ju1_2)):
            if shu_ju1_2.loc[m,'资产名称'] == shu_ju2_2.loc[i,'资产名称']:
                shu_ju2_2.loc[i, '资产编号'] = shu_ju1_2.loc[m + count, '资产编号']
                count+=1
                break

    #删除表1和表2不需要的行(暂定条件：1、资产编码为空则删除该行 2、表1的最后一行也要删除)
    shu_ju1_2.drop(index=(len(shu_ju1_2)-1),axis=0,inplace=True)#先删除表1的最后一行

    for i in range(1,len(shu_ju1_2)):
        cell = shu_ju1_2.loc[i,'复合数量']
        f_cell = float(cell)
        if np.isnan(f_cell) == True:
            shu_ju1_2.drop(index=i, axis=0, inplace=True)

    for i in range(1, len(shu_ju2_2)):
        cell = shu_ju2_2.loc[i, '数量']
        f_cell = float(cell)
        if np.isnan(f_cell) == True:
            shu_ju2_2.drop(index=i, axis=0, inplace=True)

    # 合并两个表格
    sheet_merge = pd.merge(shu_ju1_2,shu_ju2_2, on=['资产编号','资产名称','规格型号','计量单位'], how='left')

    # 参考原值保留3位小数
    for i in range(1,len(sheet_merge)):
        sheet_merge.loc[i,'参考原值'] = format(sheet_merge.loc[i,'参考原值'],'.2f')

    # 获取项目名称
    all_name = shu_ju2_2.loc[4,'安装地点']# 该单元格内容为“宁209H23平台地面集输工程（场站）”
    project_name = all_name[:all_name.find('(')]# 获取该单元格内容到“（”为止

    # 获取弃置费用
    cost_of_disposal = float(shu_ju2_9.loc[0,'Unnamed: 6'])

    # 在sheet_merge所有列的最后添加默认值的属性列
    num_col_merge = len(sheet_merge.columns) # 计算sheet_merge已有列数
    sheet_merge.insert(num_col_merge, '项目名称',project_name) # 在所有列的末尾添加

    num_col_merge = len(sheet_merge.columns) # 计算sheet_merge已有列数
    sheet_merge.insert(num_col_merge, '自编号',project_name) # 在所有列的末尾添加

    # 删除sheet_merge中的多余行
    for i in range(0, len(sheet_merge)):
        cell = sheet_merge.loc[i, '资产编号']
        f_cell = float(cell)
        if np.isnan(f_cell) == True:
            sheet_merge.drop(index=i, axis=0, inplace=True)

    # 在sheet_merge末尾添加一行“弃置费用（场站）”
    last_asset_no = sheet_merge.iloc[-1]['资产编号']# 获取最后一行的资产编号
    new_asset_no = f"{int(last_asset_no.split('_')[0]) + 1}_1"# 获取新的资产编号

    # 将新行添加到sheet_merge的末尾
    new_row = {}
    sheet_merge = sheet_merge.append(new_row, ignore_index=True)
    # 定义新行的内容
    sheet_merge.loc[len(sheet_merge)-1,'资产编号'] = new_asset_no
    sheet_merge.loc[len(sheet_merge)-1,'资产名称'] = '弃置费用（场站）'
    sheet_merge.loc[len(sheet_merge)-1,'交付日期'] = sheet_merge.loc[len(sheet_merge)-2,'交付日期']
    sheet_merge.loc[len(sheet_merge)-1,'制造厂家'] = sheet_merge.loc[len(sheet_merge)-2,'制造厂家']
    sheet_merge.loc[len(sheet_merge)-1,'规格型号'] = project_name
    sheet_merge.loc[len(sheet_merge)-1,'项目名称'] = project_name
    sheet_merge.loc[len(sheet_merge)-1,'自编号'] = project_name
    sheet_merge.loc[len(sheet_merge)-1,'计量单位'] = '座'
    sheet_merge.loc[len(sheet_merge)-1,'存放(安装地点)'] = '-'
    sheet_merge.loc[len(sheet_merge)-1,'参考原值'] = cost_of_disposal
    sheet_merge.loc[len(sheet_merge)-1,'参考原值'] = format(sheet_merge.loc[len(sheet_merge)-1, '参考原值'], '.2f')
    sheet_merge.loc[len(sheet_merge)-1,'复合数量'] = '1'

    # 输出合并后的表格
    sheet_merge.to_excel('D:\intelligent transfer of capital\data of transfer of capital\sheet_merge.xlsx',index=False)

    print('data processing finished!!!')

# Data_Processing(shu_ju1_1,shu_ju2_1,shu_ju2_9)
import pandas as pd
import numpy as np
from collections import defaultdict
from heapq import heappush, heappop


# 读取Excel文件
def read_excel(file_path):
    return pd.read_excel(file_path)


# 计算语素相似度
def morpheme_similarity(word1, word2):
    set1 = set(word1)
    set2 = set(word2)
    common_morphemes = len(set1 & set2)
    total_length = len(set1) + len(set2)
    return 2 * common_morphemes / total_length if total_length > 0 else 0


# 计算字序相似度
def order_similarity(word1, word2):
    pos_map1 = {char: idx for idx, char in enumerate(word1)}
    pos_map2 = {char: idx for idx, char in enumerate(word2)}
    common_chars = set(pos_map1.keys()) & set(pos_map2.keys())
    inversions = sum(1 for i in common_chars for j in common_chars
                     if i != j and pos_map1[i] < pos_map1[j] and pos_map2[i] > pos_map2[j])
    return 1 / (1 + inversions) if common_chars else 0


# 计算词长相似度
def length_similarity(word1, word2):
    len_diff = abs(len(word1) - len(word2))
    len_sum = len(word1) + len(word2)
    return len_diff / len_sum if len_sum > 0 else 0


# 综合计算两个词汇的语义相似度
def semantic_similarity(word1, word2):
    ms = morpheme_similarity(word1, word2)
    os = order_similarity(word1, word2)
    ls = length_similarity(word1, word2)
    # 线性回归模型公式
    return 0.7 * ms + 0.29 * os + 0.01 * ls


# 计算综合相似度
def compute_similarity(asset1, asset2):
    name_sim = semantic_similarity(str(asset1['资产名称']), str(asset2['资产名称']))
    category_sim = semantic_similarity(str(asset1['资产类别编码']), str(asset2['资产类别编码']))
    group_sim = semantic_similarity(str(asset1['资产组编码']), str(asset2['资产组编码']))
    return 0.6 * name_sim + 0.3 * category_sim + 0.1 * group_sim


# 计算相似度矩阵
def compute_similarity_matrix(data):
    n = len(data)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(data.iloc[i], data.iloc[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
    return matrix


# 平均链接法计算新相似度
def average_linkage(new_cluster, other_cluster, similarity_matrix):
    new_indices = [i for i, cluster in enumerate(clusters) if cluster == new_cluster]
    other_indices = [i for i, cluster in enumerate(clusters) if cluster == other_cluster]
    total_sim = sum(similarity_matrix[i, j] for i in new_indices for j in other_indices)
    return total_sim / (len(new_indices) * len(other_indices))


# 更新相似度矩阵
def update_matrix(similarity_matrix, clusters, new_cluster):
    n = len(clusters)
    new_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if clusters[i] != new_cluster and clusters[j] != new_cluster:
                new_matrix[i, j] = similarity_matrix[clusters.index(clusters[i]), clusters.index(clusters[j])]
                new_matrix[j, i] = similarity_matrix[clusters.index(clusters[j]), clusters.index(clusters[i])]
    return new_matrix


# 凝聚型层次聚类算法
def agglomerative_clustering(data, threshold=0.85):
    n = len(data)
    global clusters
    clusters = [[i] for i in range(n)]
    similarity_matrix = compute_similarity_matrix(data)
    priority_queue = []

    # 初始化优先队列
    for i in range(n):
        for j in range(i + 1, n):
            heappush(priority_queue, (-similarity_matrix[i, j], i, j))

    while priority_queue:
        sim, i, j = heappop(priority_queue)
        sim = -sim
        if sim < threshold:
            break
        new_cluster = clusters[i] + clusters[j]
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(new_cluster)

        # 更新相似度矩阵和优先队列
        similarity_matrix = update_matrix(similarity_matrix, clusters, new_cluster)
        for k in range(len(clusters)):
            if k != len(clusters) - 1:
                new_sim = average_linkage(new_cluster, clusters[k], similarity_matrix)
                heappush(priority_queue, (-new_sim, len(clusters) - 1, k))

    return clusters


# 处理聚类结果并输出到Excel
def process_clusters(data, clusters):
    result = []
    for cluster in clusters:
        merged_assets = ', '.join([str(data.iloc[i]['资产名称']) for i in cluster])
        for i in cluster:
            row = data.iloc[i].tolist() + [merged_assets]
            result.append(row)
    return pd.DataFrame(result, columns=['资产名称', '资产类别编码', '资产组编码', '由哪些资产合并而来'])


# 主函数
def main():
    file_path = 'asset_name_category_group.xlsx'
    data = read_excel(file_path)
    clusters = agglomerative_clustering(data)
    result = process_clusters(data, clusters)
    result.to_excel('clustered_assets.xlsx', index=False)


if __name__ == "__main__":
    main()
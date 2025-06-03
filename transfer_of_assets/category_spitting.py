import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import numpy as np


# 读取Excel文件
def read_excel(file_path):
    return pd.read_excel(file_path)


# 提取TF-IDF特征向量
def extractTFIDFVectors(data, column):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data[column])
    return X.toarray()


# 计算属性差异度μ
def computeAttributeVariance(X):
    std_dev = np.std(X)
    max_val = np.max(X)
    if max_val == 0:
        return 0
    return std_dev / max_val


# 找到最近的簇
def findNearestCluster(cluster, all_clusters):
    min_distance = float('inf')
    nearest_cluster = None
    for other_cluster in all_clusters:
        if cluster is not other_cluster:
            distance = cosine(np.mean(cluster, axis=0), np.mean(other_cluster, axis=0))
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = other_cluster
    return nearest_cluster


# 资产类别拆分算法
def assetCategorySplitting(data, column):
    X = extractTFIDFVectors(data, column)
    mu = computeAttributeVariance(X)

    if mu < 0.5:
        return [data]

    N = len(data)
    k = max(N // 80, 2)
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(X)
    clusters = [data[kmeans.labels_ == i] for i in range(k)]

    final_clusters = []
    for cluster in clusters:
        if len(cluster) > 100:
            sub_clusters = assetCategorySplitting(cluster, column)
            final_clusters.extend(sub_clusters)
        elif len(cluster) < 3:
            nearest = findNearestCluster(extractTFIDFVectors(cluster, column),
                                         [extractTFIDFVectors(c, column) for c in clusters if c is not cluster])
            if nearest is not None:
                index = clusters.index(nearest)
                clusters[index] = pd.concat([clusters[index], cluster])
        else:
            final_clusters.append(cluster)

    return final_clusters


# 主函数
def main():
    file_path = 'asset_name_specification.xlsx'
    data = read_excel(file_path)

    # 拆分类别
    clusters = assetCategorySplitting(data, '规格型号')

    # 输出到新的Excel文件
    with pd.ExcelWriter('split_assets.xlsx') as writer:
        for i, cluster in enumerate(clusters):
            cluster.to_excel(writer, sheet_name=f'Cluster_{i + 1}', index=False)


if __name__ == "__main__":
    main()
from neo4j import GraphDatabase
import pandas as pd
from Levenshtein import distance as levenshtein_distance

# Neo4j 数据库连接配置
URI = "http://localhost:7474"  #  Neo4j 地址
USER = "neo4j"                  # 用户名
PASSWORD = "******"           # 密码,此处省略

# 相似度阈值
SIM_THRESHOLD = 0.8

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def fetch_kg(session, query):
    """从 Neo4j 数据库中获取三元组集合"""
    result = session.run(query)
    return [f"{record['entity']}-{record['relation']}-{record['attribute']}" for record in result]

def calculate_similarity(pi, qj):
    """计算两个字符串之间的相似度"""
    edit_distance = levenshtein_distance(pi, qj)
    length_sum = len(pi) + len(qj)
    similarity = 1 - (edit_distance / length_sum) if length_sum > 0 else 0
    return similarity

def match_and_save_results(p_set, q_set, matched_file, unmatched_file):
    """匹配 P 和 Q 中的元素，并保存结果到 Excel 文件"""
    matched_data = []
    unmatched_p = []
    unmatched_q = []

    for pi in p_set:
        matched = False
        for qj in q_set:
            sim = calculate_similarity(pi, qj)
            if sim >= SIM_THRESHOLD:
                matched_data.append({"P": pi, "Q": qj, "Similarity": sim})
                matched = True
                break
        if not matched:
            unmatched_p.append(pi)

    # 找出未匹配的 Q
    for qj in q_set:
        if not any(d["Q"] == qj for d in matched_data):
            unmatched_q.append(qj)

    # 将匹配和未匹配的结果保存到 Excel
    matched_df = pd.DataFrame(matched_data)
    unmatched_p_df = pd.DataFrame(unmatched_p, columns=["Unmatched_P"])
    unmatched_q_df = pd.DataFrame(unmatched_q, columns=["Unmatched_Q"])

    with pd.ExcelWriter(matched_file, engine="openpyxl") as writer:
        matched_df.to_excel(writer, sheet_name="Matched", index=False)
        unmatched_p_df.to_excel(writer, sheet_name="Unmatched_P", index=False)
        unmatched_q_df.to_excel(writer, sheet_name="Unmatched_Q", index=False)

    print(f"Results saved to {matched_file}")

def main():
    # 查询语句（根据实际图谱结构修改）
    kg1_query = """
    MATCH (n)-[r]->(m)
    RETURN n.name AS entity, TYPE(r) AS relation, m.value AS attribute
    """
    kg2_query = """
    MATCH (n)-[r]->(m)
    RETURN n.name AS entity, TYPE(r) AS relation, m.value AS attribute
    """

    with driver.session() as session:
        # 获取 KG1 和 KG2 的三元组集合
        p_set = fetch_kg(session, kg1_query)
        q_set = fetch_kg(session, kg2_query)

    # 匹配并保存结果
    match_and_save_results(p_set, q_set, "matching_results.xlsx", "unmatched_results.xlsx")

    # 关闭数据库连接
    driver.close()

if __name__ == "__main__":
    main()
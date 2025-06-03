import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


def linearize_table(table):
    """将表格线性化为行列结构化表示"""
    rows = []
    for _, row in table.iterrows():
        row_data = []
        for col_name in table.columns:
            row_data.append(f"{col_name}: {row[col_name]}")
        rows.append(" | ".join(row_data))
    return rows


def filter_high_frequency_columns(table, threshold=0.3):
    """筛选高频列（非空值比例超过阈值的列）"""
    non_empty_ratios = table.notna().mean()
    high_freq_cols = non_empty_ratios[non_empty_ratios > threshold].index
    return table[high_freq_cols]


def process_wikitables():
    # 获取脚本所在目录
    script_dir = os.path.dirname(__file__)

    # 构建输入和输出目录的相对路径
    input_dir = os.path.join(script_dir, "tables_redi2_1")
    output_dir = os.path.join(script_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # 处理表格文件
    table_files = [f for f in os.listdir(input_dir) if f.startswith("re_tables-") and f.endswith(".json")]
    for table_file in tqdm(table_files, desc="Processing WikiTables"):
        table_path = os.path.join(input_dir, table_file)
        with open(table_path, 'r') as f:
            table_data = json.load(f)

        # 转换为DataFrame
        table_df = pd.DataFrame(table_data)

        # 筛选高频列
        filtered_table = filter_high_frequency_columns(table_df)

        # 线性化表格
        linearized_rows = linearize_table(filtered_table)

        # 保存处理后的表格
        output_table_path = os.path.join(output_dir, f"processed_{table_file}")
        with open(output_table_path, 'w') as f:
            json.dump(linearized_rows, f)

    # 处理查询文件
    query_file = os.path.join(script_dir, "queries.txt")
    with open(query_file, 'r') as f:
        queries = f.readlines()

    processed_queries = [q.strip() for q in queries]
    with open(os.path.join(output_dir, "processed_queries.txt"), 'w') as f:
        f.writelines(processed_queries)

    # 处理表格-查询关系文件
    rel_file = os.path.join(script_dir, "qtrels.txt")
    rel_df = pd.read_csv(rel_file, sep=' ', names=["query_id", "0", "table_id", "relevance"])

    # 保留相关性大于0的记录
    relevant_rel = rel_df[rel_df["relevance"] > 0][["query_id", "table_id"]]

    relevant_rel.to_csv(os.path.join(output_dir, "processed_qtrels.txt"), sep=' ', index=False, header=False)


if __name__ == "__main__":
    process_wikitables()
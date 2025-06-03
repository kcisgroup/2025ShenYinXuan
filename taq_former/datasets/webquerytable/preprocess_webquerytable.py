import os
import pandas as pd
import re
import html
from html.parser import HTMLParser
import numpy as np
from tqdm import tqdm
import json


class MLStripper(HTMLParser):
    """用于移除HTML标签的辅助类"""

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = ""

    def handle_data(self, d):
        self.text += d

    def get_data(self):
        return self.text.strip()


def strip_tags(html_content):
    """移除HTML标签"""
    stripper = MLStripper()
    stripper.feed(html.unescape(html_content))
    return stripper.get_data()


def clean_table_content(table):
    """清理表格内容中的HTML标签和特殊字符"""
    cleaned_table = {}
    for key, value in table.items():
        if isinstance(value, str):
            cleaned_value = strip_tags(value)
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value)
            cleaned_table[key] = cleaned_value
        else:
            cleaned_table[key] = value
    return cleaned_table


def process_webquerytable():
    # 获取脚本所在目录
    script_dir = os.path.dirname(__file__)

    # 构建输入和输出目录的相对路径
    input_dir = script_dir
    output_dir = os.path.join(script_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # 处理表格文件
    table_file = os.path.join(input_dir, "WQT.dataset.table.tsv")
    table_df = pd.read_csv(table_file, sep='\t', names=["table_id", "table_json"])

    processed_tables = []
    for _, row in tqdm(table_df.iterrows(), desc="Processing Tables"):
        table_id = row["table_id"]
        table_json = json.loads(row["table_json"])

        # 清理表格内容
        cleaned_table = clean_table_content(table_json)

        processed_tables.append({
            "table_id": table_id,
            "cleaned_table": json.dumps(cleaned_table)
        })

    processed_table_df = pd.DataFrame(processed_tables)
    processed_table_df.to_csv(os.path.join(output_dir, "processed_tables.tsv"), sep='\t', index=False)

    # 处理查询文件
    query_file = os.path.join(input_dir, "WQT.dataset.query.tsv")
    query_df = pd.read_csv(query_file, sep='\t', names=["query_id", "query_text"])

    query_df["cleaned_query"] = query_df["query_text"].apply(lambda x: strip_tags(x))
    query_df[["query_id", "cleaned_query"]].to_csv(os.path.join(output_dir, "processed_queries.tsv"), sep='\t',
                                                   index=False)

    # 处理查询-表格关系文件
    rel_file = os.path.join(input_dir, "WQT.dataset.query-table.tsv")
    rel_df = pd.read_csv(rel_file, sep='\t', names=["query_id", "table_id", "relevance"])

    rel_df.to_csv(os.path.join(output_dir, "processed_qtrels.tsv"), sep='\t', index=False, header=False)


if __name__ == "__main__":
    process_webquerytable()
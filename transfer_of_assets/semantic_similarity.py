import pandas as pd

# ----------- 相似度计算函数 -----------

def morpheme_similarity(word1, word2):
    set1, set2 = set(word1), set(word2)
    common = set1 & set2
    return 2 * len(common) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0

def inversion_number(seq1, seq2):
    common = [c for c in seq1 if c in seq2]
    mapped = {c: seq2.index(c) for c in common}
    reordered = [mapped[c] for c in common]
    count = 0
    for i in range(len(reordered)):
        for j in range(i + 1, len(reordered)):
            if reordered[i] > reordered[j]:
                count += 1
    return count

def order_similarity(word1, word2):
    inv = inversion_number(word1, word2)
    return 1 / (1 + inv)

def length_similarity(word1, word2):
    return abs(len(word1) - len(word2)) / (len(word1) + len(word2)) if (len(word1) + len(word2)) > 0 else 0

def semantic_similarity(word1, word2):
    mor_sim = morpheme_similarity(word1, word2)
    ord_sim = order_similarity(word1, word2)
    len_sim = length_similarity(word1, word2)
    return 0.7 * mor_sim + 0.29 * ord_sim + 0.01 * (1 - len_sim)

# ----------- 聚类函数 -----------

def group_similar_words(words, threshold=0.7):
    groups = []
    used = set()

    for word in words:
        if word in used:
            continue
        group = [word]
        used.add(word)
        for other in words:
            if other not in used and semantic_similarity(word, other) >= threshold:
                group.append(other)
                used.add(other)
        groups.append(group)
    return groups

# ----------- 主处理逻辑 -----------

def process_and_group(file_path, threshold=0.7):
    df = pd.read_excel(file_path)
    names = df.iloc[:, 0].dropna().astype(str).tolist()

    # 去重
    unique_names = list(set(names))

    # 分组
    groups = group_similar_words(unique_names, threshold)

    # 准备输出
    merged_names = []
    original_sets = []

    for group in groups:
        rep = group[0]
        merged_names.append(rep)
        original_sets.append(','.join(sorted(group)))

    out_df = pd.DataFrame({
        '合并后资产名称': merged_names,
        '原始资产名称集合': original_sets
    })

    out_df.to_excel('asset_name_grouped.xlsx', index=False)
    print("处理完成，已输出至 asset_name_grouped.xlsx")

# ----------- 执行 -----------

if __name__ == '__main__':
    process_and_group('asset_name.xlsx', threshold=0.7)

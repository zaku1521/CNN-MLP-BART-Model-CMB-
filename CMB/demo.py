import pandas as pd
import lmdb
import pickle
import torch
from rdkit import Chem
from tqdm import tqdm
import os
from sklearn.metrics import pairwise_distances
import numpy as np
import csv
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
import Levenshtein  # 引入Levenshtein库

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载SMILES与IR数据的LMDB数据库
def load_data_from_lmdb(db_path):
    """从LMDB数据库中加载SMILES和IR数据"""
    data = []
    assert os.path.isfile(db_path), f"{db_path} not found"
    env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data_point = pickle.loads(value)
            if 'smi' in data_point and 'ir' in data_point:
                data.append({
                    'smi': data_point['smi'],
                    'ir': data_point['ir']
                })
    env.close()
    return data

# 获取标准SMILES
def get_canonical_smile(testsmi):
    """获取标准SMILES字符串"""
    try:
        mol = Chem.MolFromSmiles(testsmi)
        if mol is None:
            print(f"Cannot convert {testsmi} to canonical smiles")
            return testsmi
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error converting {testsmi}: {str(e)}")
        return testsmi

# 加载CSV文件中的SMILES数据
def load_smiles_from_csv(csv_path):
    """从CSV文件中加载SMILES"""
    return pd.read_csv(csv_path)['SMILES'].tolist()

# 计算最大公共子结构(MCS)的Tanimoto相似度
def calculate_mcs(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0
    mcs = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, 1024)
    mcs2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, 1024)
    return DataStructs.TanimotoSimilarity(mcs, mcs2)

# 计算实际的莱文斯坦距离（编辑操作数）
def calculate_levenshtein_distance(smiles1, smiles2):
    return Levenshtein.distance(smiles1, smiles2)  # 使用Levenshtein库计算实际编辑距离

# 计算Top-k准确率
def calculate_top_k_accuracy(predictions, true_smiles, k):
    count = 0
    for pred_list, true_smi in zip(predictions, true_smiles):
        if any(same_smi(pred, true_smi) for pred in pred_list[:k]):
            count += 1
    return count / len(true_smiles)

# 判断SMILES是否相同
def same_smi(smi1, smi2):
    """判断两个SMILES是否相同"""
    return get_canonical_smile(smi1) == get_canonical_smile(smi2)

# 测试并对比真实SMILES与预测SMILES
def evaluate_predictions(test_data_path, results_csv_path, smiles_output_csv_path):
    # 加载数据
    ir_data = load_data_from_lmdb(test_data_path)
    results_df = pd.read_csv(results_csv_path)
    true_smiles = load_smiles_from_csv(smiles_output_csv_path)

    # 输出IR数据并让用户选择
    print("IR数据及编号：")
    for idx, data in enumerate(ir_data):
        print(f"{idx}: {data['ir']}")

    selected_idx = int(input("请选择一个IR数据编号进行测试: "))
    selected_ir = ir_data[selected_idx]['ir']
    selected_true_smiles = true_smiles[selected_idx]
    print(f"\n【该条数据集中的标准SMILES】\n{selected_true_smiles}\n")

    # 获取对应的预测SMILES
    rank_smiles = results_df.iloc[selected_idx, 1:].values.tolist()

    # 新增：逐条打印rank1~rank10
    print("【该条数据预测得到的SMILES（rank1 ~ rank10）】")
    for i, smi in enumerate(rank_smiles, start=1):
        print(f"rank{i}: {smi}")
    print()  # 空行美观一点

    # 计算Top-1, Top-5, Top-10准确率
    top_1_accuracy = calculate_top_k_accuracy([rank_smiles], [selected_true_smiles], 1) * 100  # 百分比
    top_5_accuracy = calculate_top_k_accuracy([rank_smiles], [selected_true_smiles], 5) * 100  # 百分比
    top_10_accuracy = calculate_top_k_accuracy([rank_smiles], [selected_true_smiles], 10) * 100  # 百分比

    # 计算最大公共子结构(MCS)的相似度（用rank1对比）
    mcs_score = calculate_mcs(selected_true_smiles, rank_smiles[0])

    # 计算莱文斯坦距离（用rank1对比）
    levenshtein_score = calculate_levenshtein_distance(selected_true_smiles, rank_smiles[0])

    # 输出结果
    print("【对比结果】")
    print(f"Top-1 Accuracy: {top_1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top_5_accuracy:.2f}%")
    print(f"Top-10 Accuracy: {top_10_accuracy:.2f}%")
    print(f"Maximum Common Substructure (MCS) Similarity: {mcs_score:.4f}")
    print(f"Levenshtein Distance: {levenshtein_score}")

# 测试阶段
test_data_path = "train.small.lmdb"  # 你需要指定测试数据的路径
results_csv_path = "results9.csv"    # 预测的SMILES结果
smiles_output_csv_path = "smiles_output.csv"  # 真实的SMILES数据

evaluate_predictions(test_data_path, results_csv_path, smiles_output_csv_path)

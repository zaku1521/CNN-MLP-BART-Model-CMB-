import lmdb
import pickle
import os
import torch
from rdkit import Chem
from functools import lru_cache
from torch.utils.data import Dataset
from torch import nn
import matplotlib.pyplot as plt
import json
import math
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import time
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
import csv
from tqdm import tqdm
import gc
import warnings
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置数据库路径
db_path = "train.lmdb"
assert os.path.isfile(db_path), f"{db_path} not found"
# 使用上下文管理器打开LMDB数据库
try:
    env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        count = txn.stat()["entries"]
        print("数据量:", count)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data = pickle.loads(value)
            print(f"Key: {key}")
            print(f"Data: {data}")
            if 'smi' in data:
                print(f"SMILES: {data['smi']}")
            if 'ir' in data:
                print(f"IR Data: {data['ir']}")
            break  # 只显示第一条记录
finally:
    env.close()
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
class IRDataset(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), f"{self.db_path} not found"
        self.env = None1
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        env.close()
    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if save_to_self:
            self.env = env
        return env
    def __len__(self):
        return len(self._keys)
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.env is None:
            self.connect_db(self.db_path, save_to_self=True)
        key = self._keys[idx]
        datapoint_pickled = self.env.begin().get(key)
        data = pickle.loads(datapoint_pickled)
        if "smi" in data:
            data["smi"] = get_canonical_smile(data["smi"])
        return data
    def __del__(self):
        if self.env is not None:
            self.env.close()
def draw_ir(ir):
    """绘制红外数据曲线"""
    plt.figure(figsize=(10, 6))
    x = ir[:, 0]
    y = ir[:, 1]
    plt.plot(x, y)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Transmittance')
    plt.title('IR Spectrum')
    plt.grid(True)
    plt.savefig('ir_spectrum.png')
    plt.close()
class MyCollator:
    def __init__(self, **kwargs):
        self.tokenizer = kwargs.get("tokenizer")
        assert self.tokenizer is not None, "Tokenizer must be provided"
        self.max_length = kwargs.get("max_length", 512)
    def __call__(self, examples):
        input_dict = {}
        smi = []
        ir = []
        for example in examples:
            if "smi" in example:
                smi.append(example["smi"])
            ir.append(example["ir"][:, 1])
        if smi:
            output = self.tokenizer(
                smi,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_dict["labels"] = output["input_ids"][:, 1:].to(device)
            input_dict["decoder_input_ids"] = output["input_ids"][:, :-1].to(device)

        input_dict["ir"] = torch.tensor(ir, dtype=torch.float32).to(device)
        return input_dict
class Linear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, bias: bool = True, init: str = "default"):
        super().__init__(d_in, d_out, bias=bias)
        self.use_bias = bias
        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)
        init_functions = {
            "default": lambda: self._trunc_normal_init(1.0),
            "relu": lambda: self._trunc_normal_init(2.0),
            "glorot": self._glorot_uniform_init,
            "gating": lambda: self._zero_init(self.use_bias),
            "normal": self._normal_init,
            "final": lambda: self._zero_init(False),
            "jax": self._jax_init
        }
        init_func = init_functions.get(init)
        if init_func is None:
            raise ValueError(f"Invalid init method: {init}")
        init_func()
    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale ** 0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)
    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)
    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)
    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
    def _jax_init(self):
        input_size = self.weight.shape[-1]
        std = math.sqrt(1 / input_size)
        nn.init.trunc_normal_(self.weight, std=std, a=-2.0 * std, b=2.0 * std)
# 添加CNN模块
class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(60)  # 输出固定大小
        )
    def forward(self, x):
        return self.conv_layers(x)
class MLP(nn.Module):
    def __init__(self, d_in, n_layers, d_hidden, d_out, activation=nn.ReLU(), bias=True, final_init="final"):
        super().__init__()
        layers = [Linear(d_in, d_hidden, bias), activation]
        for _ in range(n_layers):
            layers += [Linear(d_hidden, d_hidden, bias), activation]
        layers.append(Linear(d_hidden, d_out, bias, init=final_init))
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
# 修改MolecularGenerator类
class MolecularGenerator(nn.Module):
    def __init__(self, config_json_path, tokenizer_path):
        super().__init__()
        with open(config_json_path, "r") as f:
            config = json.load(f)
            self.model = BartForConditionalGeneration(config=BartConfig(**config))
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        # 添加CNN层
        self.cnn = CNN(in_channels=1, out_channels=32)
        # 修改MLP输入维度以匹配CNN输出
        self.mlp = MLP(32 * 60, 3, 512, 768, activation=nn.ReLU())
        # 添加批归一化层
        self.batch_norm = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)
    def ir_forward(self, ir):
        # 调整输入维度为 [batch_size, channels, sequence_length]
        x = ir.unsqueeze(1)  # 添加通道维度
        # CNN处理
        x = self.cnn(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        # 展平CNN输出
        x = x.flatten(1)
        # MLP处理
        x = self.mlp(x)
        # 调整形状以匹配BART期望的输入
        x = x.unsqueeze(1).expand(-1, 50, -1)
        return x
    def forward(self, **kwargs):
        ir = kwargs.pop("ir")
        ir_embedding = self.ir_forward(ir)
        return self.model(inputs_embeds=ir_embedding, **kwargs)

    @torch.no_grad()
    def infer(self, num_beams=10, num_return_sequences=None, max_length=512, **kwargs):
        ir_embedding = self.ir_forward(kwargs.pop("ir"))
        result = self.model.generate(
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences or num_beams,
            inputs_embeds=ir_embedding,
            decoder_start_token_id=0,
        )
        smiles = [
            self.tokenizer.decode(i, skip_special_tokens=True).strip()
            for i in result
        ]
        return smiles
    def load_weights(self, path):
        if path is not None:
            state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
            self.load_state_dict(state_dict)
def main(
        model_path=None,
        config_json_path="bart.json",
        tokenizer_path="tokenizer-smiles-bart/",
        model_weight=None,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        weight_decay=0,
        num_train_epochs=3,  # 设置默认训练轮数
        max_train_steps=None,  # 改为None，由epoch计算
        gradient_accumulation_steps=1,
        lr_scheduler_type="linear",
        num_warmup_steps=0,
        output_dir="./data",
        seed=42,
        block_size=512,
):
    # 参数验证和目录创建
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    # 初始化模型
    model = MolecularGenerator(config_json_path, tokenizer_path)
    if model_weight:
        model.load_weights(model_weight)
    model = model.to(device)
    # 准备数据集
    train_dataset = IRDataset("train.lmdb")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=MyCollator(tokenizer=model.tokenizer, max_length=block_size),
        batch_size=per_device_train_batch_size,
    )
    # 准备优化器
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # 计算总训练步数
    num_update_steps_per_epoch = len(train_dataloader)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    # 准备学习率调度器
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    # 训练循环
    total_steps = 0
    total_loss = 0
    print(f"Starting training for {num_train_epochs} epochs, {max_train_steps} total steps")
    for epoch in range(num_train_epochs):
        model.train()
        epoch_loss = 0
        # 使用tqdm创建进度条
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_train_epochs}",
            total=len(train_dataloader)
        )
        for step, batch in enumerate(progress_bar):
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            epoch_loss += loss.item()
            # 反向传播
            loss.backward()
            # 更新参数
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_steps += 1
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{epoch_loss / (step + 1):.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            # if total_steps >= max_train_steps:
            #     break
        # 每个epoch结束后保存模型
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}-cnn.pt")
        )
        print(f"Epoch {epoch + 1}: Average loss = {epoch_loss / len(train_dataloader):.4f}")
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
    # 保存最终模型
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "final_model_cnn.pt")
    )
    return total_loss / total_steps
def run_test(test_data_path, weights_path, batch_size, output_csv):
    """运行测试函数"""
    assert os.path.exists(test_data_path), f"Test data path {test_data_path} not found"
    assert os.path.exists(weights_path), f"Weights path {weights_path} not found"

    # 初始化模型
    model = MolecularGenerator("bart.json", "tokenizer-smiles-bart")
    model.load_weights(weights_path)
    model.to(device)
    model.eval()

    # 准备测试数据
    test_dataset = IRDataset(test_data_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MyCollator(tokenizer=model.tokenizer, max_length=512),
    )
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index'] + [f'rank{i}' for i in range(1, 11)])
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
                predictions = model.infer(
                    num_beams=10,
                    num_return_sequences=10,
                    **batch
                )
                for i in range(len(predictions) // 10):
                    writer.writerow(
                        [batch_idx * batch_size + i] + predictions[i * 10:(i + 1) * 10]
                    )
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
if __name__ == "__main__":

    #### 测试阶段可以选择关掉训练的代码
    # # 训练
    avg_loss = main(
        num_train_epochs=20,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        output_dir="./outputs"
    )
    print(f"Training completed with average loss: {avg_loss:.4f}")
    # 测试

    # warnings.filterwarnings(action="ignore")
    # 设置测试数据路径、模型权重路径、批次大小和输出文件路径

    run_test(
        test_data_path="train.small.lmdb",
        weights_path="./outputs/checkpoint-epoch-30-cnn.pt",
        batch_size=16,
        output_csv="results.csv"
    )
def get_InchiKey(smi):
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        return None
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
        return key
    except:
        return None
def judge_InchiKey(key1, key2):
    if key1 is None or key2 is None:
        return False
    return key1 == key2
def same_smi(smi1, smi2):
    key1 = get_InchiKey(smi1)
    if key1 is None:
        return False
    key2 = get_InchiKey(smi2)
    if key2 is None:
        return False
    return judge_InchiKey(key1, key2)

ground_truth_smiles = []
# 设置数据库路径
db_path = "train.small.lmdb"  # 确保替换为你的实际lmdb文件名
assert os.path.isfile(db_path), f"{db_path} not found"
# 打开LMDB数据库
env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin() as txn:
    count = txn.stat()["entries"]
    print("数据量:", count)
with env.begin() as txn:
    # 获取数据库中的所有键
    cursor = txn.cursor()
    for key, value in cursor:
        # 反序列化值
        data = pickle.loads(value)
        # 打印键和值的内容
        print(f"Key: {key}")
        print(f"Data: {data}")
        # 根据数据内容可以进行不同的处理
        if 'smi' in data:
            print(f"SMILES: {data['smi']}")
            ground_truth_smiles.append(data['smi'])
        if 'ir' in data:
            print(f"IR Data: {data['ir']}")
def evaluate_model(test_data_path, weights_path, batch_size):
    current_dir = os.getcwd()
    config_json_path = os.path.join(current_dir, "bart.json")
    tokenizer_path = os.path.join(current_dir, "tokenizer-smiles-bart")

    model = MolecularGenerator(
        config_json_path=config_json_path,
        tokenizer_path=tokenizer_path,
    )
    model.load_weights(weights_path)
    model.cuda()
    model.eval()

    test_data = DataLoader(
        IRDataset(
            db_path=test_data_path,
        ),
        shuffle=False,
        collate_fn=MyCollator(tokenizer=model.tokenizer, max_length=512),
        batch_size=batch_size,
        drop_last=False,
    )


    top_1_count = 0
    top_3_count = 0
    top_5_count = 0
    top_10_count = 0
    total_count = 0



    with torch.no_grad():
        for idx, i in tqdm(enumerate(test_data)):
            print("====idx==", idx)
            total_count += 1
            result = model.infer(tokenizer=model.tokenizer, length_penalty=0, num_beams=10, **i)
            # ground_truth_smiles = [data["smi"] for data in i["smi"]] if "smi" in i else []
            if same_smi(ground_truth_smiles[idx], result[0]):
                top_1_count += 1
            # 使用循环来检查前3个元素
            found_in_top_3 = False
            for j in range(min(3, len(result))):
                if same_smi(ground_truth_smiles[idx], result[j]):
                    found_in_top_3 = True
                    break
            if found_in_top_3:
                top_3_count += 1
            # 使用循环来检查前5个元素
            found_in_top_5 = False
            for z in range(min(5, len(result))):
                if same_smi(ground_truth_smiles[idx], result[z]):
                    found_in_top_5 = True
                    break
            if found_in_top_5:
                top_5_count += 1
            # 使用循环来检查所有元素
            found_in_top_10 = False
            for t in range(len(result)):
                if same_smi(ground_truth_smiles[idx], result[t]):
                    found_in_top_10 = True
                    break
            if found_in_top_10:
                top_10_count += 1
            # if idx > 1:  # 这里可以根据需要调整测试数据的量
            #     break

    top_1_accuracy = top_1_count / total_count
    top_3_accuracy = top_3_count / total_count
    top_5_accuracy = top_5_count / total_count
    top_10_accuracy = top_10_count / total_count
    weighted_score = 0.4 * top_1_accuracy + 0.1 * top_3_accuracy + 0.1 * top_5_accuracy + 0.4 * top_10_accuracy
    print(f"Top-1 Accuracy: {top_1_accuracy}")
    print(f"Top-3 Accuracy: {top_3_accuracy}")
    print(f"Top-5 Accuracy: {top_5_accuracy}")
    print(f"Top-10 Accuracy: {top_10_accuracy}")
    print(f"Weighted Score: {weighted_score}")
    return weighted_score

# 使用示例
current_dir = os.getcwd()
test_data_path = os.path.join(current_dir, "train.small.lmdb")
weights_path = os.path.join(current_dir, "./outputs/checkpoint-epoch-30-cnn.pt")
batch_size = 1
evaluate_model(test_data_path, weights_path, batch_size)

class ModelValidator:
    def __init__(self, model, val_dataloader, device):
        self.model = model
        self.val_dataloader = val_dataloader
        self.device = device
    def validate_epoch(self):
        """验证一个epoch的模型性能"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                # 计算准确率
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                mask = labels != -100
                correct = (predictions[mask] == labels[mask]).sum().item()
                total_correct += correct
                total_samples += mask.sum().item()
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy
        }

class ModelEvaluator:
    def __init__(self, model, test_dataloader, device):
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        results = {
            'test_loss': 0,
            'accuracy': 0,
            'predictions': [],
            'ground_truth': []
        }
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                outputs = self.model(**batch)
                # 收集预测结果
                predictions = self.model.infer(num_beams=10, **batch)
                results['predictions'].extend(predictions)
                # 如果有ground truth，收集它们
                if 'labels' in batch:
                    ground_truth = [
                        self.model.tokenizer.decode(label, skip_special_tokens=True)
                        for label in batch['labels']
                    ]
                    results['ground_truth'].extend(ground_truth)
        # 计算评估指标
        if results['ground_truth']:
            results['accuracy'] = self.calculate_accuracy(
                results['predictions'],
                results['ground_truth']
            )
        return results

    @staticmethod
    def calculate_accuracy(results_file, ground_truth):
        # Initialize counters for top-k accuracies
        tops = [1, 3, 5, 10]
        correct_counts = {k: 0 for k in tops}
        total = 0

        # Create ground_truth dictionary mapping
        truth_dict = {i: smi for i, smi in enumerate(ground_truth)}

        with open(results_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header row

            for row in csv_reader:
                try:
                    idx = int(row[0])
                    predictions = row[1:]  # Get all 10 predictions

                    # Check if idx exists in ground truth
                    if idx not in truth_dict:
                        print(f"Warning: Index {idx} not found in ground truth data")
                        continue

                    true_smi = truth_dict[idx]
                    total += 1

                    # Calculate accuracy for each top-k
                    for k in tops:
                        for pred in predictions[:k]:
                            if same_smi(pred, true_smi):
                                correct_counts[k] += 1
                                break

                except Exception as e:
                    print(f"Error processing row {row}: {str(e)}")
                    continue

        # Print results
        print("\nAccuracy Results:")
        print(f"Total valid samples: {total}")
        accuracies = {}

        for k in tops:
            accuracy = correct_counts[k] / total if total > 0 else 0
            accuracies[f'top_{k}'] = accuracy
            print(f"Top-{k} Accuracy: {accuracy:.4f}")

        return accuracies
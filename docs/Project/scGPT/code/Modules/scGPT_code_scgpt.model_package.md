# Modules

## scgpt.model package

### scgpt.model.dsbn module
- Domain-Specific Batch Normalization for Unsupervised Domain Adaptation (DSBN)
- from "Pytorch implementation of Domain-Specific Batch Normalization for Unsupervised Domain Adaptation" (CVPR2019).


```py
class scgpt.model.dsbn.DomainSpecificBatchNorm1d(
    num_features: int, num_domains: int, eps: float = 1e-05, momentum: float = 0.1,
    affine: bool = True, track_running_stats: bool = True)
```
```py
class scgpt.model.dsbn.DomainSpecificBatchNorm2d(
    num_features: int, num_domains: int, eps: float = 1e-05, momentum: float = 0.1, 
    affine: bool = True, track_running_stats: bool = True)
```
### scgpt.model.generation_model module

#### .ClsDecoder
用于分类任务的解码器模块

```py
d_model：整数，表示输入和输出特征的维度 (解码器中的每一个线性层的输入和输出维度都保持一致)
n_cls：整数，表示分类任务中的类别数
nlayers：整数，默认为3，指定解码器中线性层的数量
activation：激活函数，默认为 nn.ReLU

输出：分类

class scgpt.model.generation_model.ClsDecoder(
    d_model: int, 
    n_cls: int, 
    nlayers: int = 3, 
    activation: callable = <class 'torch.nn.modules.activation.ReLU'>)

```

#### .GeneEncoder

将离散的符号转换为连续的向量表示（embedding)

输入:

- 类型: Tensor（张量）
- 形状: [batch_size, sequence_length]
- 内容: 这个张量包含整数，代表序列中每个元素的索引。在自然语言处理中，这些通常是词汇的索引；在其他应用，如基因序列分析中，这些可能代表特定的基因或编码标记。

输出:

- 类型: Tensor（张量）
- 形状: [batch_size, sequence_length, embedding_dim]
- 内容: 输出的张量包含了输入序列中每个元素的嵌入表示。每个嵌入是一个embedding_dim维的向量

```py
num_embeddings：整数，表示嵌入层中的嵌入（词汇）数量，即词汇表的大小。
embedding_dim：整数，表示每个嵌入向量的维度。
padding_idx：可选的整数，如果提供，这个索引所对应的嵌入向量会被初始化为零，并且在训练过程中不会被更新。这通常用于序列处理中的填充字符。

class scgpt.model.generation_model.GeneEncoder(
    num_embeddings: int, 
    embedding_dim: int, 
    padding_idx: int | None = None)
```

举例：
```py
# 假设的词汇表
vocab = {'hello': 0, 'world': 1, 'goodbye': 2, 'earth': 3, 'pad': 4}
padding_idx = vocab['pad']

# 构造GeneEncoder实例
encoder = GeneEncoder(num_embeddings=len(vocab), embedding_dim=10, padding_idx=padding_idx)

# 假设的输入数据（词汇索引）
sentences = torch.tensor([
    [vocab['hello'], vocab['world'], vocab['pad']],
    [vocab['goodbye'], vocab['world'], vocab['pad']]
])

# 转换数据
encoded_sentences = encoder(sentences)

print("Encoded Sentences:")
print(encoded_sentences)
print("Shape of Encoded Sentences:", encoded_sentences.shape)

```
Output:
<pre>
Encoded Sentences:
tensor([[[ 0.5006,  0.0033,  0.7780,  0.2959, -0.8926, -2.3452, -0.5962,
           0.0652,  1.1002,  1.0908],
         [ 0.0698, -0.7755, -1.2345,  0.5933,  0.7286,  2.3914, -0.8417,
          -0.2821,  0.0423, -0.6916],
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000,  0.0000,  0.0000]],

        [[ 0.0898,  0.2482, -0.4086, -1.2100, -1.4511, -1.0510,  0.3044,
           1.5130,  1.6055,  0.3598],
         [ 0.0698, -0.7755, -1.2345,  0.5933,  0.7286,  2.3914, -0.8417,
          -0.2821,  0.0423, -0.6916],
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000,  0.0000,  0.0000]]], grad_fn=<NativeLayerNormBackward0>)
Shape of Encoded Sentences: torch.Size([2, 3, 10])
</pre>

#### .PositionalEncoding
给序列中的每个元素添加位置信息，使模型能够考虑到序列中元素的顺序

输入:

- 类型: Tensor（张量）
- 形状: [seq_len, batch_size, embedding_dim]
- 内容: 其中包含了序列的嵌入表示。

输出:

- 类型: Tensor（张量）
- 形状: [seq_len, batch_size, embedding_dim]
- 内容:输入张量位置编码相加后的结果，并应用dropout。这样处理后的输出包含了原始嵌入信息和位置信息，可以被后续的模型层（如 Transformer 中的自注意力层）使用

```py
d_model: 整数，表示嵌入的维度。
dropout: 浮点数，默认为0.1，表示在位置编码添加到嵌入向量后应用的dropout比率，用于防止过拟合。
max_len: 整数，默认为5000，表示序列的最大长度，用于预先计算所有可能位置的位置编码

class scgpt.model.generation_model.PositionalEncoding(
    d_model: int, 
    dropout: float = 0.1, 
    max_len: int = 5000)
```

举例：
```py
# 假设我们有一个batch_size为1的输入序列，每个序列长度为10，嵌入维度为512
batch_size = 1
seq_len = 10
d_model = 512

encoder = PositionalEncoding(d_model)
input_tensor = torch.randn(seq_len, batch_size, d_model)

# 应用位置编码
output = encoder(input_tensor)
```

#### .Similarity

```py
temp: 表示温度参数，用于调节相似度的数值。温度参数较低时，相似度的差异会被放大，较高时则会减小差异

class scgpt.model.generation_model.Similarity(temp)
```

举例：
```py
# 假设温度参数为 0.1
similarity = Similarity(temp=0.1)

# 创建两个随机的特征向量
feature_vector1 = torch.randn(1, 10)  # 假设有1个样本，特征维度为10
feature_vector2 = torch.randn(1, 10)

# 计算两个特征向量之间的相似度
sim_value = similarity(feature_vector1, feature_vector2)
print("Similarity Value:", sim_value)
```

#### .TransformerGenerator

输入:

`src` (Tensor): 

   - 形状: `[batch_size, seq_len]`
   - 内容: 代表序列中每个位置的令牌索引，如基因或词的索引。

`values` (Tensor):

   - 形状: `[batch_size, seq_len]`
   - 内容: 通常与`src`的令牌对应，表示每个令牌的数值信息，如基因的表达级别或任何其他相关的数值特征。

`input_pert_flags` (Tensor):

   - 形状: `[batch_size, seq_len]`
   - 内容: 指示某个位置的令牌是否经历了扰动或修改，通常用于模型训练中的数据增强或特殊处理。

`src_key_padding_mask` (Tensor):

   - 形状: `[batch_size, seq_len]`
   - 内容: 一个布尔张量，用于指示哪些元素是填充元素，因此在处理时应被忽略。

输出:

输出是一个字典，包含根据传入的标志参数决定的多个张量：

`mlm_output` (Tensor):

   - 形状: `[batch_size, seq_len]`
   - 内容: 语言模型预测的输出，即预测每个位置的下一个可能的令牌。

`cls_output` (Tensor):

   - 条件: 当`CLS=True`
   - 形状: `[batch_size, n_cls]`
   - 内容: 分类任务的输出，其中`n_cls`是目标类别数。

`mvc_output` (Tensor):

   - 条件: 当`MVC=True`
   - 形状: `[batch_size, seq_len]`
   - 内容: 掩码值预测的输出，用于预测可能被掩码或扰动的值。

`loss_ecs` (Tensor):

   - 条件: 当`ECS=True`
   - 形状: 标量
   - 内容: 弹性单元相似度目标的损失值，用于优化模型以增强相似样本之间的相似度。

举例：
```py
import torch

# 假设词汇表和必要的配置已经设置
vocab = {'<pad>': 0, '<cls>': 1, 'gene1': 2, 'gene2': 3, ...}
ntoken = len(vocab)

# 初始化模型
model = TransformerGenerator(
    ntoken=ntoken,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    nlayers_cls=2,
    n_cls=10,
    vocab=vocab,
    dropout=0.1,
    pad_token='<pad>',
    pad_value=vocab['<pad>'],
    pert_pad_id=vocab['<cls>'],
    do_mvc=True,
    domain_spec_batchnorm=False,
    cell_emb_style='cls',
    mvc_decoder_style='inner product',
    ecs_threshold=0.3,
    explicit_zero_prob=False,
    use_fast_transformer=False,
    pre_norm=True
)

# 准备输入数据
src = torch.randint(0, ntoken, (10, 5))  # 10个样本，序列长度5
values = torch.randn(10, 50)  # 每个基因的表达值
input_pert_flags = torch.zeros(10, 50, dtype=torch.long)  # 无扰动
src_key_padding_mask = (src == vocab['<pad>'])

# 前向传递
output = model(src, values, input_pert_flags, src_key_padding_mask, CLS=True, MVC=True, ECS=True)
print(output)  # 打印输出结果

```


```py
scgpt.model.generation_model.TransformerGenerator(
    ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, nlayers_cls: int, n_cls: int, 
    vocab: Any, dropout: float = 0.5, pad_token: str = '<pad>', pad_value: int = 0, pert_pad_id: int = 2, 
    do_mvc: bool = False, domain_spec_batchnorm: bool | str = False, cell_emb_style: str = 'cls', 
    mvc_decoder_style: str = 'inner product', ecs_threshold: float = 0.3, explicit_zero_prob: bool = False, 
    use_fast_transformer: bool = False, fast_transformer_backend: str = 'flash', pre_norm: bool = False)
```

#### .generate_square_subsequent_mask

用于生成一个特定大小的方阵，这个方阵在对角线上和对角线以下的位置为0，在对角线以上的位置为负无穷大（-inf）

e.g., 确保在预测序列中的每个步骤时，模型只能看到先前的序列信息



```py
sz (int)：掩码矩阵的大小，即生成的方阵的行数和列数

scgpt.model.generation_model.generate_square_subsequent_mask(sz: int)→ Tensor
```

举例：
```py
import torch

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

# 生成一个大小为5的序列掩码
mask = generate_square_subsequent_mask(5)
print(mask
```

Output:
<pre>
tensor([[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]])
</pre>

### scgpt.model.grad_reverse module

```py

```

### scgpt.model.model module

#### .AdversarialDiscriminator

假设我们有一个数据集，其中包含不同批次生产的生物样本的基因表达数据。我们想要使用 AdversarialDiscriminator 来训练一个模型，以消除批次效应，使得数据在进行后续分析如聚类或分类时更加准确

输入：

- 类型：Tensor
- 形状：[batch_size, emb_size]
- 描述：输入数据为一个张量，它代表一批数据，其中每个数据项都有 emb_size 个特征。这个形状表明每次处理的是一个批量的数据，batch_size 指定了每个批次中的数据项数量，而 emb_size 对应 d_model，即每项数据的维度或特征数量。

输出：

- 类型：Tensor
- 形状：[batch_size, n_cls]
- 描述：输出也是一个张量，其形状为 [batch_size, n_cls]。这表示对于输入的每个数据项，模型将输出一个向量，该向量的维度由 n_cls 确定。在默认设置中，n_cls 为1，这常用于二分类问题（比如，判断输入是否属于某个类别）。每个输出的元素都对应于输入数据的一个预测类别概率或得分

```py
d_model: 每一层的维度，默认为 3。
n_cls: 输出层的类别数，默认为 1，通常用于二分类问题。
activation: 激活函数，默认为 LeakyReLU。
reverse_grad: 是否反转梯度，默认为 False，在对抗训练中用于训练生成器。

如果将 reverse_grad 设置为 True，则该类会在某些层应用梯度反转技术，这有助于模型学习到不受批次特性影响的特征，从而改进模型在不同批次数据上的表现。

class .AdversarialDiscriminator(
    d_model: int, 
    n_cls: int, 
    nlayers: int = 3, 
    activation: callable = <class 'torch.nn.modules.activation.LeakyReLU'>, 
    reverse_grad: bool = False
    )
```

举例：
```py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 假设的数据和批次标签
data_tensor = torch.randn(100, 3)  # 100个样本，每个样本3个特征
batch_labels = torch.randint(0, 2, (100,))  # 假设有两个批次0和1

# 创建 DataLoader
dataset = TensorDataset(data_tensor, batch_labels)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 实例化模型
model = AdversarialDiscriminator(d_model=3, n_cls=2, activation=nn.LeakyReLU, reverse_grad=True)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):  # 训练10个周期
    for data, target in dataloader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# 假设测试数据如下
test_data = torch.randn(20, 3)
test_labels = torch.randint(0, 2, (20,))

# 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(test_data)
    predicted = torch.argmax(test_outputs, dim=1)
    accuracy = (predicted == test_labels).float().mean()
    print(f"Accuracy: {accuracy.item()}")

```

#### .BatchLabelEncoder

将离散的生物学数据（如基因、蛋白质或其他生物标志物的索引）转换为连续的嵌入向量，以便在神经网络或其他机器学习模型中使用。这种转换可以捕捉到生物学数据中的隐含特征，使得模型能够更好地进行分类、预测或其他分析任务

输入：

- 形状（Shape）：(batch_size, sequence_length)
- batch_size：批次的大小，即一次处理的样本数量。
- sequence_length：每个样本中的特征数量。
- 内容：每个元素是一个整数，代表生物学数据的索引。例如，如果我们有一个包含 100 个基因的词汇表，那么索引的取值范围是 0 到 99。

输出：

- 形状（Shape）：(batch_size, sequence_length, embedding_dim)
- batch_size：批次的大小，与输入相同。
- sequence_length：每个样本中的特征数量，与输入相同。
- embedding_dim：嵌入向量的维度。
- 内容：每个元素是一个浮点数，代表嵌入向量的一个维度的值。这个值是经过嵌入和层归一化处理的结果，表示输入索引对应的特征向量。


```py
class .BatchLabelEncoder(
    num_embeddings: int, 
    embedding_dim: int, 
    padding_idx: int | None = None)
```

举例：

```py
import torch
from torch import nn, Tensor
from typing import Optional

# 初始化模型
num_embeddings = 5  # 基因词汇表大小
embedding_dim = 3   # 嵌入向量维度
model = BatchLabelEncoder(num_embeddings, embedding_dim)

# 输入数据
# 假设我们有 3 个样本，每个样本包含 4 个基因的索引
input_data = torch.tensor([
    [0, 1, 2, 3],
    [3, 2, 1, 0],
    [4, 4, 4, 4]
])

# 进行前向传播
output = model(input_data)
print(output)
```

<pre>
示例输出:
	•	输出的形状为 (3, 4, 3)，即批次大小为 3，每个样本的序列长度为 4，嵌入向量维度为 3。
	•	每个元素是一个浮点数，代表经过嵌入和层归一化处理后的基因索引的特征向量。

tensor([
    [[ 0.1, -0.2,  0.3],
     [ 0.0,  0.2,  0.1],
     [-0.1,  0.1,  0.2],
     [ 0.3, -0.3,  0.0]],

    [[ 0.3, -0.3,  0.0],
     [-0.1,  0.1,  0.2],
     [ 0.0,  0.2,  0.1],
     [ 0.1, -0.2,  0.3]],

    [[ 0.2,  0.2,  0.2],
     [ 0.2,  0.2,  0.2],
     [ 0.2,  0.2,  0.2],
     [ 0.2,  0.2,  0.2]]
])
</pre>


#### .CategoryValueEncoder

输入：

- 形状（Shape）：(batch_size, seq_len)
- batch_size：批次的大小，即一次处理的样本数量。
- seq_len：每个样本中的特征数量（序列长度）。
- 内容：每个元素是一个整数，代表生物学数据的索引。

输出：

- 形状（Shape）：(batch_size, seq_len, embedding_dim)
- batch_size：批次的大小，与输入相同。
- seq_len：每个样本中的特征数量，与输入相同。
- embedding_dim：嵌入向量的维度。
- 内容：每个元素是一个浮点数，代表嵌入向量的一个维度的值。这个值是经过嵌入和层归一化处理的结果。


```py
class .CategoryValueEncoder(
    num_embeddings: int, 
    embedding_dim: int, 
    padding_idx: int | None = None)
```
举例：

假设我们有一个基因表达数据集，每个样本用基因的索引表示，我们希望将这些索引转换为嵌入向量

```py

import torch
from torch import nn, Tensor
from typing import Optional

# 初始化模型
num_embeddings = 5  # 基因词汇表大小
embedding_dim = 3   # 嵌入向量维度
model = CategoryValueEncoder(num_embeddings, embedding_dim)

# 输入数据
# 假设我们有 3 个样本，每个样本包含 4 个基因的索引
input_data = torch.tensor([
    [0, 1, 2, 3],
    [3, 2, 1, 0],
    [4, 4, 4, 4]
])

# 进行前向传播
output = model(input_data)
print(output)
```

*与BatchLabelEncoder代码十分相似*


#### .ClsDecoder

ClsDecoder 类是一个用于分类任务的解码器，常用于神经网络模型的最后一层，将特征向量转换为类别预测。在生物信息学领域，这种解码器可以用于将嵌入向量或特征表示转换为具体的生物学分类结果，如基因表达数据的分类、蛋白质功能预测等

输入:

- 形状（Shape）：(batch_size, embsize)
- batch_size：批次的大小，即一次处理的样本数量。
- embsize：嵌入向量或特征向量的维度，等同于 d_model。
- 内容：每个元素是一个浮点数，代表样本的特征向量。

输出:

- 形状（Shape）：(batch_size, n_cls)
- batch_size：批次的大小，与输入相同。
- n_cls：分类任务中的类别数。
- 内容：每个元素是一个浮点数，表示该样本属于每个类别的分数。

```py
class .ClsDecoder(
    d_model: int, 
    n_cls: int, 
    nlayers: int = 3, 
    activation: callable = <class 'torch.nn.modules.activation.ReLU'>)
```

举例：

假设我们有一个基因表达数据集，每个样本用特征向量表示，并且我们希望将这些特征向量转换为类别预测

```py
import torch
from torch import nn, Tensor
from typing import Optional

# 初始化模型
d_model = 4   # 输入特征向量的维度
n_cls = 3     # 分类任务中的类别数
model = ClsDecoder(d_model, n_cls)

# 输入数据
# 假设我们有 2 个样本，每个样本的特征向量维度为 4：
input_data = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8]
])

# 进行前向传播
output = model(input_data)
print(output)

```
输出：
<pre>
	•	输出的形状为 (2, 3)，即批次大小为 2，每个样本的类别数为 3。
	•	每个元素是一个浮点数，表示该样本属于每个类别的分数。
tensor([
    [0.25, -0.10, 0.35],
    [0.45,  0.20, 0.55]
])
</pre>

#### .ContinuousValueEncoder

ContinuousValueEncoder 类是一个用于将连续数值编码为向量的神经网络投影层，在生物信息学领域中可以用于将连续的生物学数据（如基因表达水平、蛋白质浓度等）转换为神经网络可以处理的特征向量。

输入:

- 形状（Shape）：(batch_size, seq_len)
- batch_size：批次的大小，即一次处理的样本数量。
- seq_len：每个样本中的特征数量（序列长度）。
- 内容：每个元素是一个浮点数，代表连续的生物学数据（如基因表达水平）。

输出:

- 形状（Shape）：(batch_size, seq_len, d_model)
- batch_size：批次的大小，与输入相同。
- seq_len：每个样本中的特征数量，与输入相同。
- d_model：编码向量的维度。
- 内容：每个元素是一个浮点数，表示每个输入连续数值对应的特征向量。


```py
d_model：编码向量的维度。
dropout：dropout 层的概率，用于防止过拟合。
max_value：连续数值的最大值，用于数值剪裁。


class .ContinuousValueEncoder(
    d_model: int, 
    dropout: float = 0.1, 
    max_value: int = 512
    )
```
举例:

假设我们有一个基因表达数据集，每个样本用连续数值表示，我们希望将这些数值转换为特征向量。

```py
import torch
from torch import nn, Tensor
from typing import Optional

# 初始化模型
d_model = 4   # 编码向量的维度
model = ContinuousValueEncoder(d_model)

# 输入数据
# 假设我们有 2 个样本，每个样本包含 3 个基因的表达水平
input_data = torch.tensor([
    [1.0, 2.5, 3.3],
    [4.0, 5.1, 6.2]
])

# 进行前向传播
output = model(input_data)
print(output)

```

输出：

<pre>
    输出的形状为 (2, 3, 4)，即批次大小为 2，每个样本的序列长度为 3，编码向量维度为 4。
    每个元素是一个浮点数，表示每个输入连续数值对应的特征向量。

tensor([
    [[ 0.5, -0.1,  0.3,  0.2],
     [ 0.7,  0.1,  0.4, -0.1],
     [ 0.8, -0.2,  0.2,  0.5]],

    [[ 0.4,  0.3, -0.3,  0.1],
     [ 0.6, -0.4,  0.5,  0.2],
     [ 0.9,  0.2, -0.1,  0.3]]
])

</pre>

#### .ExprDecoder

ExprDecoder 类是一个用于解码表达值的神经网络模块，可能用于基因表达数据或其他连续值预测任务。这个解码器支持两种功能：基础预测和显式的零概率预测，这在处理稀疏数据（如基因表达数据）时非常有用

输入:

- 形状（Shape）：(batch_size, seq_len, d_model)
- batch_size：批次的大小，即一次处理的样本数量。
- seq_len：每个样本中的特征数量（序列长度）。
- d_model：输入特征向量的维度。
- 内容：每个元素是一个浮点数，代表特征向量。

输出:

- 形状（Shape）：
    - 如果 explicit_zero_prob 为 False：
        - {'pred': (batch_size, seq_len)}
    - 如果 explicit_zero_prob 为 True：
        - {'pred': (batch_size, seq_len), 'zero_probs': (batch_size, seq_len)}
- 内容：每个元素是一个浮点数，表示每个输入特征向量对应的预测值和零概率。



```py
d_model：输入特征向量的维度。
explicit_zero_prob：是否计算显式的零概率。
use_batch_labels：是否使用批次标签，影响输入的维度。

class .ExprDecoder(
    d_model: int, 
    explicit_zero_prob: bool = False, 
    use_batch_labels: bool = False)
```

举例:

假设我们有一个基因表达数据集，每个样本用特征向量表示，我们希望将这些特征向量转换为基因表达值预测，并且在某些情况下需要显式计算零概率。

```py
import torch
from torch import nn, Tensor
from typing import Dict

# 初始化模型
d_model = 4   # 输入特征向量的维度
model = ExprDecoder(d_model, explicit_zero_prob=True)

# 输入数据
# 假设我们有 2 个样本，每个样本包含 3 个特征向量，维度为 4
input_data = torch.tensor([
    [[1.0, 0.5, -1.2, 0.3], [0.8, -0.6, 1.5, -0.7], [0.5, -1.2, 0.4, 1.1]],
    [[0.9, -0.8, 1.2, -0.5], [1.1, 0.7, -1.4, 0.6], [0.4, 0.3, -0.9, 1.2]]
])

# 进行前向传播
output = model(input_data)
print(output)

```
输出：

<pre>
输出的 pred 形状为 (2, 3)，表示每个样本和每个特征向量的预测值。
输出的 zero_probs 形状为 (2, 3)，表示每个样本和每个特征向量的零概率。

{
    'pred': tensor([
        [0.35, 0.45, 0.30],
        [0.40, 0.50, 0.35]
    ]),
    'zero_probs': tensor([
        [0.20, 0.15, 0.25],
        [0.10, 0.05, 0.30]
    ])
}

</pre>


#### .FastTransformerEncoderWrapper

FastTransformerEncoderWrapper 类是一个封装器，用于构建和使用快速 Transformer 编码器（Fast Transformer Encoder）。这种编码器可以在生物信息学领域中应用，例如处理基因序列、蛋白质序列或其他高维度数据的建模任务。

输入:

- 形状（Shape）：
    - src：形状为 (N, seq_len, embsize) 的张量。
        - N：批次大小。
        - seq_len：序列长度。
        - embsize：嵌入向量的维度。
    - src_key_padding_mask：形状为 (N, seq_len) 的布尔张量。
- 每个元素为布尔值，表示对应位置是否是填充标记。

输出:

- 形状（Shape）：输出张量的形状为 (N, seq_len, embsize)，与输入形状相同。
- 内容：每个元素是一个浮点数，表示经过快速 Transformer 编码器处理后的特征表示。

```py
d_model：输入嵌入的维度。
nhead：多头注意力机制中的头数。
d_hid：前馈神经网络中的隐藏层维度。
nlayers：编码器的层数。
dropout：dropout 的概率，用于防止过拟合。

class .FastTransformerEncoderWrapper(
    d_model: int, 
    nhead: int, 
    d_hid: int, 
    nlayers: int, 
    dropout: float = 0.5)
```

举例：

```py
# 初始化模型
d_model = 4   # 输入嵌入的维度
nhead = 2     # 多头注意力机制中的头数
d_hid = 8     # 前馈神经网络中的隐藏层维度
nlayers = 2   # 编码器的层数
dropout = 0.1 # dropout 概率

model = FastTransformerEncoderWrapper(d_model, nhead, d_hid, nlayers, dropout)

# 输入数据
# 假设我们有 2 个样本，每个样本包含 3 个特征向量，维度为 4：
input_data = torch.tensor([
    [[1.0, 0.5, -1.2, 0.3], [0.8, -0.6, 1.5, -0.7], [0.5, -1.2, 0.4, 1.1]],
    [[0.9, -0.8, 1.2, -0.5], [1.1, 0.7, -1.4, 0.6], [0.4, 0.3, -0.9, 1.2]]
])
padding_mask = torch.tensor([
    [False, False, True],
    [False, True, False]
])

# 进行前向传播
output = model(input_data, padding_mask)
print(output)

```

结果：
<pre>
输出的形状为 (2, 3, 4)，即批次大小为 2，每个样本的序列长度为 3，嵌入维度为 4。
每个元素是一个浮点数，表示经过快速 Transformer 编码器处理后的特征表示。

tensor([
    [[ 0.1,  0.2, -0.1,  0.3],
     [ 0.4, -0.2,  0.5, -0.3],
     [ 0.2,  0.1,  0.4,  0.1]],

    [[ 0.3, -0.1,  0.2, -0.2],
     [ 0.2,  0.4, -0.1,  0.5],
     [ 0.1,  0.3, -0.2,  0.4]]
])

</pre>

#### .FlashTransformerEncoderLayer

FlashTransformerEncoderLayer 类是一个改进的 Transformer 编码器层，**支持 FlashAttention**，可以在生物信息学领域中应用，例如处理基因序列、蛋白质序列或其他高维度数据的建模任务。这个类继承自 nn.Module，并且修改了标准的 torch.nn.TransformerEncoderLayer 以支持 FlashAttention。

输入:

- 形状（Shape）：
    - src：形状为 (batch_size, seq_len, embsize) 的张量（假设 batch_first=True）。
        - batch_size：批次大小。
        - seq_len：序列长度。
        - embsize：特征维度（d_model）。
    - src_mask：可选，输入序列的掩码。
    - src_key_padding_mask：可选，形状为 (batch_size, seq_len) 的布尔张量。

输出:

- 形状（Shape）：输出张量的形状为 (batch_size, seq_len, embsize)，与输入形状相同。
- 内容：每个元素是一个浮点数，表示经过编码器层处理后的特征表示


```py
d_model：输入特征的维度。
nhead：多头注意力机制中的头数。
dim_feedforward：前馈神经网络的维度，默认为 2048。
dropout：dropout 的概率，默认为 0.1。
activation：中间层的激活函数，默认为 ReLU，可以选择 ReLU 或 GELU。
layer_norm_eps：层归一化的 eps 值，默认为 1e-5。
batch_first：如果为 True，则输入和输出张量的形状为 (batch, seq, feature)，默认为 True。
norm_scheme：归一化方案，"pre" 或 "post"。

class .FlashTransformerEncoderLayer(
    d_model, nhead, dim_feedforward=2048, 
    dropout=0.1, activation='relu', layer_norm_eps=1e-05, 
    batch_first=True, device=None, dtype=None, norm_scheme='post')
```
举例：

```py

# 初始化模型
d_model = 4   # 输入特征的维度
nhead = 2     # 多头注意力机制中的头数
dim_feedforward = 8  # 前馈神经网络的维度
dropout = 0.1 # dropout 概率

model = FlashTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

# 输入数据
input_data = torch.tensor([
    [[1.0, 0.5, -1.2, 0.3], [0.8, -0.6, 1.5, -0.7], [0.5, -1.2, 0.4, 1.1]],
    [[0.9, -0.8, 1.2, -0.5], [1.1, 0.7, -1.4, 0.6], [0.4, 0.3, -0.9, 1.2]]
])
padding_mask = torch.tensor([
    [False, False, True],
    [False, True, False]
])

# 进行前向传播
output = model(input_data, src_key_padding_mask=padding_mask)
print(output)
```

输出：

<pre>
输出的形状为 (2, 3, 4)，即批次大小为 2，每个样本的序列长度为 3，特征维度为 4。
每个元素是一个浮点数，表示经过编码器层处理后的特征表示。
tensor([
    [[ 0.1,  0.2, -0.1,  0.3],
     [ 0.4, -0.2,  0.5, -0.3],
     [ 0.2,  0.1,  0.4,  0.1]],

    [[ 0.3, -0.1,  0.2, -0.2],
     [ 0.2,  0.4, -0.1,  0.5],
     [ 0.1,  0.3, -0.2,  0.4]]
])

</pre>


#### .GeneEncoder

同scgpt.model.generation_model.GeneEncoder

```py
class .GeneEncoder(
    num_embeddings: int, 
    embedding_dim: int, 
    padding_idx: int | None = None)
```

#### .MVCDecoder

MVCDecoder 类是一个用于细胞嵌入的掩码值预测解码器。在生物信息学领域，特别是在单细胞 RNA 测序（scRNA-seq）数据分析中，这个解码器可以用于从细胞嵌入向量预测基因表达值。

输入:

- 形状（Shape）：
    - cell_emb：形状为 (batch, embsize=d_model) 的张量。
    - gene_embs：形状为 (batch, seq_len, embsize=d_model) 的张量。
- 内容：
    - cell_emb：每个元素是一个浮点数，表示细胞嵌入向量。
    - gene_embs：每个元素是一个浮点数，表示基因嵌入向量。

输出:

- 形状（Shape）：
    - 如果 explicit_zero_prob 为 False：
        - {'pred': (batch, seq_len)}
    - 如果 explicit_zero_prob 为 True：
    - {'pred': (batch, seq_len), 'zero_probs': (batch, seq_len)}
- 内容：
    - pred：每个元素是一个浮点数，表示预测的基因表达值。
    - zero_probs：每个元素是一个浮点数，表示预测的零概率（如果启用


```py
d_model：基因嵌入的维度。
arch_style：解码器的架构风格，有以下几种选择：
"inner product"
"inner product, detach"
"concat query"
"sum query"
query_activation：用于查询向量的激活函数，默认是 nn.Sigmoid。
hidden_activation：用于隐藏层的激活函数，默认是 nn.PReLU。
explicit_zero_prob：是否显式计算零概率。
use_batch_labels：是否使用批次标签。

class .MVCDecoder(
    d_model: int, arch_style: str = 'inner product', query_activation: ~torch.nn.modules.module.Module = <class 'torch.nn.modules.activation.Sigmoid'>, 
    hidden_activation: ~torch.nn.modules.module.Module = <class 'torch.nn.modules.activation.PReLU'>, explicit_zero_prob: bool = False, use_batch_labels: bool = False)
```

```py
# 初始化模型
d_model = 4   # 基因嵌入的维度
arch_style = "inner product"  # 解码器的架构风格
model = MVCDecoder(d_model, arch_style)

# 输入数据
cell_emb = torch.tensor([[0.5, 1.2, -0.8, 0.3], [1.1, -0.7, 0.6, 0.2]])  # shape (2, 4)
gene_embs = torch.tensor([
    [[0.9, 0.1, -1.0, 0.4], [0.8, -0.2, 1.1, -0.6], [0.5, -1.2, 0.4, 1.0]],
    [[0.6, -0.5, 0.7, -0.1], [1.2, 0.8, -1.3, 0.7], [0.3, 0.4, -0.8, 1.1]]
])  # shape (2, 3, 4)

# 进行前向传播
output = model(cell_emb, gene_embs)
print(output)
```

<pre>>
输出的 pred 形状为 (2, 3)，表示每个样本和每个基因嵌入向量的预测值
{
    'pred': tensor([
        [0.35, 0.45, 0.30],
        [0.40, 0.50, 0.35]
    ])
}
</pre>

#### .PositionalEncoding

同scgpt.model.generation_model.PositionalEncoding

```py
class .PositionalEncoding(
    d_model: int, 
    dropout: float = 0.1, 
    max_len: int = 5000)
```
#### Similarity
同scgpt.model.generation_model.Similarity

```py
class Similarity(temp)
```
#### .TransformerModel

TransformerModel 类是一个基于 Transformer 的模型，设计用于处理和分析生物信息学领域的数据，特别是单细胞 RNA 测序（scRNA-seq）数据。该模型集成了多种功能模块，用于不同的任务，如基因表达预测、分类、对比学习等。

```

ntoken：词汇表的大小。
d_model：嵌入向量的维度。
nhead：多头注意力机制中的头数。
d_hid：前馈神经网络中的隐藏层维度。
nlayers：Transformer 编码器的层数。
nlayers_cls：分类解码器的层数，默认为 3。
n_cls：分类任务中的类别数，默认为 1。
vocab：词汇表对象，用于查找 padding token 的索引。
dropout：dropout 的概率，默认为 0.5。
pad_token：填充标记，默认为 <pad>。
pad_value：填充值，默认为 0。
do_mvc：是否启用 MVC 解码器。
do_dab：是否启用 DAB 模块。
use_batch_labels：是否使用批标签。
num_batch_labels：批标签的数量。
domain_spec_batchnorm：是否使用特定域的 batchnorm。
input_emb_style：输入嵌入风格，默认为 "continuous"。
n_input_bins：输入 bin 的数量（如果使用分类输入嵌入风格）。
cell_emb_style：细胞嵌入风格，默认为 "cls"。
mvc_decoder_style：MVC 解码器风格，默认为 "inner product"。
ecs_threshold：ECS 任务的阈值。
explicit_zero_prob：是否显式计算零概率。
use_fast_transformer：是否使用快速 Transformer。
fast_transformer_backend：快速 Transformer 的后端，默认为 "flash"。
pre_norm：是否使用 pre-norm 方案。

class model.TransformerModel(
    ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, nlayers_cls: int = 3, n_cls: int = 1, vocab: Any | None = None, dropout: float = 0.5, pad_token: str = '<pad>', pad_value: int = 0, do_mvc: bool = False, do_dab: bool = False, use_batch_labels: bool = False, num_batch_labels: int | None = None, domain_spec_batchnorm: bool | str = False, input_emb_style: str = 'continuous', n_input_bins: int | None = None, cell_emb_style: str = 'cls', mvc_decoder_style: str = 'inner product', ecs_threshold: float = 0.3, explicit_zero_prob: bool = False, use_fast_transformer: bool = False, fast_transformer_backend: str = 'flash', pre_norm: bool = False)
```

```py
# 初始化模型
ntoken = 10  # 词汇表大小
d_model = 4  # 嵌入向量的维度
nhead = 2    # 多头注意力机制中的头数
d_hid = 8    # 前馈神经网络的维度
nlayers = 2  # Transformer 编码器的层数
vocab = {"<pad>": 0}

model = TransformerModel(ntoken, d_model, nhead, d_hid, nlayers, vocab=vocab)

# 输入数据
# 假设我们有 2 个样本，每个样本包含 3 个基因序列和相应的值，维度为 4
src = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
values = torch.tensor([[0.5, 0.8, 0.2], [0.6, 0.7, 0.9]])  # shape (2, 3)
src_key_padding_mask = torch.tensor([[False, False, True], [False, True, False]])  # shape (2, 3)

# 进行前向传播
output = model(src, values, src_key_padding_mask)
print(output)
```

<pre>
输出的 mlm_output 形状为 (2, 3)，表示每个样本和每个序列元素的预测值。
输出的 cell_emb 形状为 (2, 4)，表示每个样本的细胞嵌入向量

{
    'mlm_output': tensor([
        [0.35, 0.45, 0.30],
        [0.40, 0.50, 0.35]
    ]),
    'cell_emb': tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ])
}

</pre>

#### generate_square_subsequent_mask

同scgpt.model.generation_model.generate_square_subsequent_mask

```py
class .generate_square_subsequent_mask(sz: int)→ Tensor
```

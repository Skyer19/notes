## scgpt.tasks.cell_emb module

### .cell_emb.embed_data

embed_data 函数用于对 AnnData 对象进行预处理，并使用指定的模型对数据进行嵌入。这在单细胞 RNA 测序数据分析中非常有用，可以生成每个细胞的嵌入表示，用于后续的分析。

Preprocess anndata and embed the data using the model.


```py

adata_or_file (Union[AnnData, PathLike])：输入的 AnnData 对象或 AnnData 文件的路径。
model_dir (PathLike)：模型目录的路径，包含词汇文件、模型配置文件和预训练模型文件。
gene_col (str, 默认值: "feature_name")：AnnData 对象中包含基因名称的列名。
max_length (int, 默认值: 1200)：输入序列的最大长度。
batch_size (int, 默认值: 64)：推理时的批处理大小。
obs_to_save (Optional[list], 默认值: None)：要保存在输出 AnnData 对象中的观测列列表。
device (Union[str, torch.device], 默认值: "cuda")：使用的设备（例如 "cuda" 或 "cpu"）。
use_fast_transformer (bool, 默认值: True)：是否使用快速 Transformer 实现（例如 flash-attn）。
return_new_adata (bool, 默认值: False)：是否返回新的 AnnData 对象。如果为 False，将嵌入结果添加到现有 AnnData 对象的 obsm 属性中。

返回值:

AnnData：带有细胞嵌入的 AnnData 对象。如果 return_new_adata 为 False，嵌入结果会添加到现有 AnnData 对象的 obsm 属性中，键为 "X_scGPT"。

scgpt.tasks.cell_emb.embed_data(
        adata_or_file: AnnData | str | PathLike, 
        model_dir: str | PathLike, 
        : str = 'feature_name', 
        max_length=1200, 
        batch_size=64, 
        obs_to_save: list | None = None, 
        device: str | device = 'cuda', 
        use_fast_transformer: bool = True, 
        return_new_adata: bool = False
    )→ AnnData
```

### .cell_emb.get_batch_cell_embeddings

Get the cell embeddings for a batch of cells.

Parameters:

- adata (AnnData) – The AnnData object.
- cell_embedding_mode (str) – The mode to get the cell embeddings. Defaults to “cls”.
- model (TransformerModel, optional) – The model. Defaults to None.
- vocab (GeneVocab, optional) – The vocabulary. Defaults to None.
- max_length (int) – The maximum length of the input sequence. Defaults to 1200.
- batch_size (int) – The batch size for inference. Defaults to 64.
- model_configs (dict, optional) – The model configurations. Defaults to None.
- gene_ids (np.ndarray, optional) – The gene vocabulary ids. Defaults to None.
- use_batch_labels (bool) – Whether to use batch labels. Defaults to False.

Returns:

- The cell embeddings.

Return type:

- np.ndarray

```py
scgpt.tasks.cell_emb.get_batch_cell_embeddings(
        adata, 
        cell_embedding_mode: str = 'cls', 
        model=None, 
        vocab=None, 
        max_length=1200, 
        batch_size=64, 
        model_configs=None, 
        gene_ids=None, 
        use_batch_labels=False
    )→ ndarray

```
### .grn.GeneEmbedding

GeneEmbedding 类用于处理基因嵌入数据，并提供各种工具进行分析和可视化。这个类在生物信息学领域尤其有用，用于分析基因相似性、生成基因网络、计算元基因分数等。

```py
classscgpt.tasks.grn.GeneEmbedding(embeddings: Mapping)
```

Methods:

```py
static average_vector_results(vec1, vec2, fname)
```
```py
# 将聚类定义转化为 DataFrame
cluster_definitions_as_df(top_n=20)[source]
```
```py
# 计算基因相似性
compute_similarities(gene, subset=None, feature_type=None)
```
```py
# 生成基因网络
generate_network(threshold=0.5)
```
```py
# 生成嵌入向量
generate_vector(genes)
```
```py
# 生成加权嵌入向量
generate_weighted_vector(genes, weights)
```
```py
# 生成 AnnData 对象
get_adata(resolution=20)
```
```py
# 获取元基因
get_metagenes(gdata)
```
```py
# 获取相似基因
get_similar_genes(vector)
```
```py
# 绘制元基因 UMAP 图
plot_metagene(gdata, mg=None, title='Gene Embedding')
```
```py
# 绘制元基因分数热图
plot_metagenes_scores(adata, metagenes, column, plot=None)
```
```py
# 绘制基因相似性条形图
plot_similarities(gene, n_genes=10, save=None)
```
```py
# 从文件中读取嵌入数据
read_embedding(filename)
```
```py
# 读取向量文件
static read_vector(vec)
```
```py
# 计算元基因分数
score_metagenes(adata, metagenes)
```


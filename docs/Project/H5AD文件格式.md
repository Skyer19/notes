H5AD 是一种文件格式，用于存储生物信息学中的单细胞表达数据。这种格式是基于 HDF5（层次数据格式版本 5）技术，HDF5 是一种用来存储大规模科学数据的通用文件格式。H5AD 特别适用于处理单细胞 RNA 测序（scRNA-seq）数据，这是一种用于在单个细胞层面上研究基因表达的技术。

H5AD 文件通常包含以下主要组件：

1. **表达矩阵**：这是 H5AD 文件的核心，包含有关各个基因在各个细胞中的表达水平的数据。
2. **变量（基因）元数据**：提供关于文件中包含的基因的额外信息，如基因名称或基因功能注释。
3. **观测（细胞）元数据**：提供有关单个细胞的额外信息，如细胞类型、实验条件或样本来源。
4. **非矩阵数据**：可以包括聚类结果、维度缩减坐标（例如 t-SNE 或 UMAP）、细胞轨迹分析结果等。

H5AD 格式被广泛应用于生物信息学软件和数据分析工具中，特别是在使用单细胞 RNA 测序数据进行高级分析时。例如，`Scanpy` 这一流行的 Python 库就支持读取和写入 H5AD 文件，以便于进行复杂的数据分析和可视化。这种格式的目的是为了实现高效的数据存储、快速访问和便于交换，从而支持生物医学研究中单细胞层面的复杂数据分析。

# 使用scanpy操作

```python
import scanpy as sc

# 加载 h5ad 文件
adata = sc.read_h5ad('mtg_subset_filtered.h5ad')

# 显示adata的简要概览
print(adata)
```

## AnnData对象

```
AnnData object with n_obs × n_vars = 4893 × 18896
    obs: 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'assay_ontology_term_id', 'cell_type_ontology_term_id', 'disease_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'organism_ontology_term_id', 'sex_ontology_term_id', 'tissue_ontology_term_id', 'is_primary_data', 'Neurotypical.reference', 'Class', 'Subclass', 'Supertype', 'Age.at.death', 'Years.of.education', 'Cognitive.status', 'ADNC', 'Braak.stage', 'Thal.phase', 'CERAD.score', 'APOE4.status', 'Lewy.body.disease.pathology', 'LATE.NC.stage', 'Microinfarct.pathology', 'Specimen.ID', 'donor_id', 'PMI', 'Number.of.UMIs', 'Genes.detected', 'Fraction.mitochrondrial.UMIs', 'suspension_type', 'development_stage_ontology_term_id', 'Continuous.Pseudo.progression.Score', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'ident', 'percent.mt', 'RNA_snn_res.0.5', 'seurat_clusters'
    var: 'vst.mean', 'vst.variance', 'vst.variance.expected', 'vst.variance.standardized', 'vst.variable', 'highly_variable', 'means', 'dispersions', 'dispersions_norm'
    uns: 'neighbors'
    obsm: 'X_pca', 'X_umap'
    varm: 'PCs'
    obsp: 'distances'
```

### 主要组成部分

**`n_obs × n_vars`**:

  - `4893 × 18896` 表示数据集中有 4893 个观测（通常是单个细胞）和 18896 个变量（通常是基因或特征）
  - 意味着对于这 4893 个单独的细胞，每个细胞都被测量了 18896 个不同的变量

**`obs`(Observations)**: 存储与观测相关的注释信息，每一行代表一个观测（细胞），每一列代表一个观测的属性

   - 这是一个 DataFrame，其中包含与每个观测（细胞）相关的元数据
   - obs 属性每行代表一个观测，通常是一个细胞。在举例的数据中，每行代表一个单独的细胞，意味着有 4893 行
   - obs 的列数并不等于 n_vars（18896），而是取决于有多少种元数据特征被记录。例如，细胞的类型、获取时间、实验条件等
   - 在举例的数据中，obs 是一个 4893 x m 的 DataFrame

**`var`(Variables)**: 存储与变量相关的信息，每行代表一个变量（基因），每列代表一个变量的属性。在举例的数据中包括：

  - `vst.mean`, `vst.variance` 等统计特征和`highly_variable` 表示是否为高变异基因
  - var 属性包含与每个变量（基因）相关的元数据。在举例的数据中，var 属性（变量或基因的元数据）有 18896 行（每行一个基因)
  - var 的列数取决于记录了多少基因特征，例如基因名称、功能、是否高度可变等
  - 在举例的数据中，var 是一个 18896 x n 的 DataFrame

**`uns`(Unstructured)**: 用来存储非结构化的数据，比如整体数据集的统计数据或元数据，如邻居图（`neighbors`）

**`obsm`(Observation matrices)**: 存储观测级别的矩阵数据，通常用于降维的结果，例如：
     - `X_pca` (主成分分析结果)
     - `X_umap` (UMAP降维结果)

**`varm`(Variable matrices)**: 存储变量级别的矩阵数据，例如主成分（`PCs`）

**`obsp`(Observation pairwise)**: 存储观测之间成对的数据，如距离矩阵（`distances`）


### 常用api

```py
# 查看变量（基因）的名称
print(adata.var_names)
```

<pre>
Output:
Index(['TSPAN6', 'TNMD', 'DPM1', 'SCYL3', 'C1orf112', 'FGR', 'CFH', 'FUCA2',
       'GCLC', 'NFYA',
       ...
       'SMIM40', 'NPBWR1', 'CDR1', 'ACTL10', 'PANO1', 'PRRC2B', 'UGT1A3',
       'UGT1A5', 'F8A2', 'F8A1'],
      dtype='object', length=18896)
</pre>

```py
# 查看观测（细胞）的名称
print(adata.obs_names)
```

<pre>
Output:
Index(['mtg_GTTACAGGTGTAGTGG-L8TX_210513_01_C09-1142430429', 
       ...
       'mtg_TAACGACCATCTCCCA-L8TX_210506_01_E08-1153814230'],
      dtype='object', length=4893)
</pre>

```py
# 查看观测的元数据
print(adata.obs)
```

<pre>
Output:
                                                       orig.ident  nCount_RNA  \
mtg_GTTACAGGTGTAGTGG-L8TX_210513_01_C09-1142430429  SeuratProject      9370.0   
...                                                           ...         ...   
mtg_TAACGACCATCTCCCA-L8TX_210506_01_E08-1153814230  SeuratProject     10144.0   

......
                                                        organism     sex  \
mtg_GTTACAGGTGTAGTGG-L8TX_210513_01_C09-1142430429  Homo sapiens  female    
...                                                          ...     ...   
mtg_TAACGACCATCTCCCA-L8TX_210506_01_E08-1153814230  Homo sapiens  female   

......                                             seurat_clusters  

mtg_GTTACAGGTGTAGTGG-L8TX_210513_01_C09-1142430429               0  
...                                                            ...   
mtg_TAACGACCATCTCCCA-L8TX_210506_01_E08-1153814230               6  

[4893 rows x 47 columns]
</pre>

```py
# 查看变量的元数据
print(adata.var)
```

<pre>
Output:
          vst.mean  vst.variance  vst.variance.expected  \
TSPAN6    0.002310      0.003428               0.002755   
 ...            ...           ...                    ...     
F8A1      0.014633      0.015318               0.020033   

          vst.variance.standardized  vst.variable  highly_variable  \
TSPAN6                     1.244589             1             True    
...                             ...           ...              ...     
F8A1                       0.764648             0            False   

                 means  dispersions  dispersions_norm  
TSPAN6    2.944253e+00     9.999864          0.828220  
...                ...          ...               ...  
F8A1      3.671453e+00     9.613371         -0.455848  

[18896 rows x 9 columns]
</pre>


### adata的.X属性

**对于基因表达数据本身，即实际的测量数据，通常存储在 AnnData 对象的 .X 属性中。** 比如在单细胞基因表达分析中，.X 通常包含细胞（行）和基因（列）的表达数据。这个矩阵可能是稠密的（如 NumPy 数组）或稀疏的（如 SciPy 稀疏矩阵），这取决于数据的性质和存储需求。

对于举例，这是一个形状为 4893 x 18896 的矩阵，其中每行代表一个细胞，每列代表一个基因的表达水平。

```py
import scanpy as sc

# 加载数据
adata = sc.read_h5ad('path/to/your_data.h5ad')

# 访问 .X 属性
data_matrix = adata.X

# 打印 .X 属性的一些信息
print(data_matrix.shape)  # 显示矩阵的维度
print(data_matrix[:5, :5])  # 打印矩阵的前5行和前5列的数据
```
<pre>
Output:
(4893, 18896)
[[-4.1185167e-02 -3.8049729e-03  2.0831785e+00 -3.7935510e-01]
 [-4.1185167e-02 -3.8049729e-03 -4.2588213e-01 -3.7935510e-01]
 [-4.1185167e-02 -3.8049729e-03  3.8974628e+00 -3.7935510e-01]
 [-4.1185167e-02 -3.8049729e-03 -4.2588213e-01  3.5818794e+00]
 [-4.1185167e-02 -3.8049729e-03 -4.2588213e-01 -3.7935510e-01]]

</pre>

**.X 属性的用途：**

`.X` 属性通常用于执行各种数据分析任务，包括但不限于：

- **数据预处理**：如标准化、对数转换等。
- **统计分析**：如计算基因的平均表达水平、变异性等。
- **降维分析**：如 PCA、t-SNE、UMAP。
- **聚类和分类**：用于细胞类型的鉴定或样本分组。

访问 `.X` 属性后，可以使用 Python 的数据分析和机器学习工具（如 Pandas, Scikit-learn, TensorFlow 等）来处理和分析这些数据。

**注意事项：**

- 如果 `.X` 存储为稀疏矩阵，在处理前可能需要转换为稠密格式，特别是在某些函数或方法不支持稀疏格式的情况下。
- 对于大型数据集，直接处理 `.X` 可能需要大量内存。在这种情况下，考虑使用分批处理或数据降维技术。

## 过滤表达矩阵

### 过滤细胞
`scanpy` 的 `sc.pp.filter_cells`函数是一个预处理函数，用于根据细胞的特性过滤细胞。这通常是基于细胞的一些计数数据，如UMI计数、检测到的基因数量等。这个函数可以帮助去除由于技术原因或样本质量问题而产生的低质量细胞。

```py
sc.pp.filter_cells(adata, min_counts=None, 
					max_counts=None, min_genes=None, max_genes=None, inplace=True)
```

**参数:**

- **adata** (`AnnData`): `AnnData` 对象，包含要过滤的数据。
- **min_counts** (`int`, 可选): 每个细胞的最小计数（总UMI计数）。低于此值的细胞将被过滤掉。
- **max_counts** (`int`, 可选): 每个细胞的最大计数。高于此值的细胞将被过滤掉，通常用于去除双细胞或污染。
- **min_genes** (`int`, 可选): 每个细胞必须表达的最小基因数。低于此数的细胞将被过滤掉，可以帮助去除损坏或空的细胞。
- **max_genes** (`int`, 可选): 每个细胞的最大基因数。高于此数的细胞可能表明样本污染或其他异常。
- **inplace** (`bool`, 默认是 `True`)

**返回值:**

- 如果 `inplace=True`，则函数没有返回值，直接修改传入的 `adata` 对象。
- 如果 `inplace=False`，则返回一个元组，其中包含两个元素，这两个返回值是：
	- cells_subset (ndarray): 一个布尔索引掩码，用于过滤细胞。True 表示保留该细胞，False 表示移除该细胞
	- number_per_cell (ndarray): Depending on what was thresholded (counts or genes), the array stores n_counts or n_cells per gene.

### 过滤基因

`scanpy` 的 `sc.pp.filter_genes`函数是一个预处理函数，用于根据基因的表达特性来过滤基因。这通常是基于基因的计数数据，如在多少个细胞中表达、总计数等。这个函数可以帮助去除不表达或低表达的基因，也可以去除异常高表达的基因，这通常是由于技术噪声或数据处理错误造成的。

```py
scanpy.pp.filter_genes(data, *, min_counts=None, 
						min_cells=None, max_counts=None, max_cells=None, inplace=True, copy=False)
```

**参数:**

- **data** (`AnnData`): 需要进行过滤的 `AnnData` 对象。
- **min_counts** (`int`, 可选): 基因在所有细胞中的最小总计数。低于此值的基因将被过滤掉。
- **min_cells** (`int`, 可选): 基因至少需要在这么多个细胞中表达。低于此数字的基因将被过滤掉。
- **max_counts** (`int`, 可选): 基因在所有细胞中的最大总计数。高于此值的基因可能会被过滤掉，用来移除异常高表达的基因。
- **max_cells** (`int`, 可选): 基因最多只能在这么多个细胞中表达。超过此数的基因可能会被过滤掉。
- **inplace** (`bool`, 默认是 `True`)

**返回值:**

- 如果 `inplace=True`，则函数没有返回值，直接修改传入的 `adata` 对象。
- 如果 `inplace=False`，则返回一个元组，其中包含两个元素，这两个返回值是：
	- gene_subset (ndarray): 一个布尔索引掩码，用于标记哪些基因被保留。其中 True 表示保留该基因，False 表示移除该基因
	- number_per_gene (ndarray): Depending on what was thresholded (counts or cells), the array stores n_counts or n_cells per gene


### 示例代码

**直接更新原来的adata：**

```python
import scanpy as sc

# 加载数据
adata = sc.read_h5ad('path/to/your_data.h5ad')

# 过滤基因，根据在至少3个细胞中有表达，且总计数至少为20
sc.pp.filter_genes(adata, min_cells=3)

# 查看过滤后的数据
print(adata)
```
**生成新的的adata：**

- 在应用sc.pp.filter_cells后进行细胞过滤
```python
# 进行细胞过滤
cell_subset, _ = sc.pp.filter_cells(adata, min_counts=500, inplace=False)

print(cell_subset.shape)

# 应用过滤掩码来更新原始adata对象
filtered_adata = adata[cell_subset].copy()

print(filtered_adata.shape)
```

- 在应用sc.pp.filter_genes后进行细胞过滤
```py
# 使用 inplace=False 进行细胞过滤，并从返回的元组中提取 AnnData 对象
gene_subset, _ = sc.pp.filter_genes(filtered_adata, min_cells=500, inplace=False)

print(gene_subset.shape)

# 应用过滤掩码来更新原始adata对象
filtered_adata = filtered_adata[:,gene_subset].copy()

print(filtered_adata)
```

### 保存为新.h5ad 文件

```py
filtered_adata.write('filtered_data.h5ad')
```




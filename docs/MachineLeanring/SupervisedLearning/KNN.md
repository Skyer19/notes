## KNN (K-nearest neighbour) classifier

KNN classifier is also called Classification with Instance-based Learning.

分类：

- One nearest neighbour
- Nearest Neighbour classifier

### One nearest neighbour

算法：

- a query sample
- Find the nearest neighbours in the data
- assgin the nearest neighbour's class label to the query sample

缺点：

- Sensitive to noise
- Overfit training data


### Nearest Neighbour classifier

算法：

- a query sample
- Find the nearest K neighbours in the data
- assgin the class label which most of m (m <=K) neighbours have to the query sample


### K 的取值
- K取奇数 -> 确保一定有一个label的数量最多

- 增加K：

	- have a smoother decision boundary (higher bias)
	- less sensitive to training data (lower variance)

- 怎么选择最适宜K：

### Distance Metrics for KNN

- Manhattan distance (L1-norm)

$$d(x_1,x_2) = \sum^{K}_{k}｜x_{1}^{k}-x_{2}^{k}｜ $$

- Euclidean distance (L2-norm)

$$d(x_1,x_2) = \sqrt{\sum^{K}_{k}｜x_{1}^{k}-x_{2}^{k}｜^{2}} $$

- Chebyshev distance (L∞-norm)

$$d(x_1,x_2) = max^{K}_{k}｜x_{1}^{k}-x_{2}^{k}｜ $$

## Distance weighted KNN

改进：

- 根据每个邻居与测试查询实例的接近程度（越接近 -> 权重越高）为每个邻居分配一个权重 $w^{(i)}$，从而优化 k-NN
- 将邻域内每个类别的权重相加，分配权重最大的类别给测试查询实例

### 常见权重分配方法

- Inverse of distance

$$ w^{(i)}=\frac{1}{d(x_1,x_2)}$$

- Gaussian distribution

$$ w^{(i)}=\frac{1}{\sqrt{2 \pi}}(-\frac{{d(x_1,x_2)^{2}}}{2})$$

注意：

- 在Distance weighted KNN 中，k 的值并不重要。距离较远的示例的权重较小，不会对分类产生很大影响
- 分类：
	- $k = N$ (size of training set): global method
	- Otherwise, local method
- 对嘈杂的训练数据具有鲁棒性：分类基于所有 k 个最近邻的加权组合，可有效消除孤立噪声的影响


**The curse of dimensionality: 随着数据特征维度的上升， 算法的性能下降**

- k-NN 依赖于距离度量，如果在高维空间中使用所有特征，则距离度量可能效果不佳
- 如果许多特征不相关，则属于同一类的实例可能彼此相距甚远
- 解决方案：对每个特征赋予不同的权重，或执行特征选择/提取

## KNN regression

- k-NN regression： 计算 k 个最近邻居的平均值
- Distance-weighted k-NN for regression： 计算 k 个最近邻的加权平均值


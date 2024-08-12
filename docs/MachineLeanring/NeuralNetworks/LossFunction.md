## 定义

## 常见损失函数

### 应用分类任务

#### 平均绝对误差（Mean Absolute Error, MAE）

定义：所有误差绝对值的平均值，表示预测值与实际值之间的平均绝对偏差。

$$MAE = \frac{1}{n}\sum^{N}_{i-1}|\hat{y}_{i} - y_{i}|$$

评判标准：
- MAE值越小，表示模型的预测结果越接近实际值。
- MAE值具有可解释性，单位与原始数据一致。

应用：适用于对误差方向不敏感且需要直接解释误差的场景

#### 均方误差（Mean Squared Error, MSE）
所有误差平方的平均值，表示预测值与实际值之间的平均平方偏差。

$$MSE = \frac{1}{n}\sum^{N}_{i-1}(\hat{y}_{i} - y_{i})^{2}$$

评判标准：
- MSE值越小，表示模型的预测结果越接近实际值
- 由于误差平方会放大大的误差，因此MSE对异常值（outliers）更加敏感

应用：适用于对大误差惩罚较重的场景，对异常值敏感

#### 均方根误差（Root Mean Squared Error, RMSE）
定义：均方误差的平方根，表示预测值与实际值之间的平均偏差

$$MSE = \sqrt{\frac{1}{n}\sum^{N}_{i-1}(\hat{y}_{i} - y_{i})^{2}}$$

评判标准：
- RMSE值越小，表示模型的预测结果越接近实际值
- RMSE与原始数据的单位一致，因此更具可解释性

应用：适用于对大误差敏感并且需要与原始数据单位一致的场景

#### R平方（R-squared, R²）

定义：衡量模型解释目标变量变异能力的指标，取值范围为0到1

评判标准：
- R²越接近1，表示模型对数据的拟合程度越好
- R²为1表示模型完美预测了所有数据点，R²为0表示模型未能解释数据中的任何变异

应用：适用于需要评估模型解释能力的场景/整体拟合效果的情况

#### 平均绝对百分比误差（Mean Absolute Percentage Error, MAPE）

定义：误差占实际值的百分比，适用于实际值不为零的情况

评判标准：
- MAPE值越小，表示模型的预测结果越接近实际值
- MAPE以百分比形式表示，因此对不同尺度的数据具有可比性

应用：适用于需要比较不同尺度数据的误差百分比的场景，但需要注意不能有实际值为零的情况

### 应用回归任务

- Binary cross - entropy:

$$L = \{frac{1}{N}\sum^{N}_{i=1}(y^{(i)}log(\hat{y}^{(i)})) + (1-(y^{(i)})log(1-\hat{y}^{(i)}))$$

- Categorical cross - entropy:

$$L = -\frac{1}{N}\sum^{N}_{i-1}\sum^{C}_{c=1}y_{c}^{(i)}log(\hat{y_{c}^{(i))}$$

$C$: 可能的标签类别

$y_{c}^{(i)$: 对于第$i$个数据，模型对于这个数据可能被分类为$c$个标签的概率

## 常见损失函数应用

- 只有一个输出并应用线性激活：回归任务 -- MSE
- 只有一个输出并应用Sigmoid激活函数-> 二分类 -- Binary cross-entropy
- 许多输出并应用Softmax激活函数-> Multi-class classification -- Categorical cross - entropy
- 许多输出并对每个输出应用Sigmoid激活函数-> Multi-label classification  -- 对于每一个应用Softmax激活函数的输出应用Binary cross - entropy



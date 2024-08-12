## 定义

$$y=\alpha x+b$$

## 损失函数
Sum of Sqaures: $E=\frac{1}{2}\sum^{N}_{i=1}(\hat{y}^{i}-y^{i})^2$

## 更新参数

$$\frac{dE}{da}=\sum^{N}_{i=1}(\hat{y}^{i}-y^{i})x^{i}$$

$$\frac{dE}{db}=\sum^{N}_{i=1}(\hat{y}^{i}-y^{i})$$

## 梯度下降

Gradient Decent:

- $\alpha$
  
$$\alpha:= \alpha:-\alpha\frac{dE}{d\alpha}}$$

- $b$

$$b:= b:-b\frac{dE}{db}}$$
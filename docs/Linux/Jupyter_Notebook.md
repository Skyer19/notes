## 在新环境中安装 Jupyter Notebook

### 激活Conda环境中安装 Jupyter Notebook
```bash
conda install jupyter
```

### 安装 ipykernel
```bash
conda install ipykernel
```

### 将新环境添加为 Jupyter Kernel
```bash
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

### 启动 Jupyter Notebook
```bash
jupyter notebook
```

## 删除环境的Jupyter Kernel
### 列出所有 Jupyter 内核
```bash
jupyter kernelspec list
```

### 删除环境对应的内核
```bash
jupyter kernelspec uninstall myenv
```

## 使用特定端口连接
```bash
ssh ssh -L 10088:localhost:10088 user@123.123.123.123
```

```bash
jupyter notebook --port 10088
```
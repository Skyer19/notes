## 使用 YAML 文件创建/更新环境

### 导出当前环境到 YAML 文件
```bash
conda env export > environment.yml
```

### 创建环境
```bash
conda env create -f environment.yml -n new_env_name
```

## 安装/使用多个 Conda 实例

### 安装
#### 下载另一个 Miniconda 安装脚本
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-second.sh
```

#### 运行安装脚本，使用批处理模式并指定安装路径
```bash
bash Miniconda3-latest.sh -b -p $HOME/miniconda3
```

```bash
bash Miniconda3-second.sh -b -p $HOME/miniconda3-second
```

#### 初始化 Conda

```bash
# 初始化第一个 Conda 实例
$HOME/miniconda3/bin/conda init

# 初始化第二个 Conda 实例
$HOME/miniconda3-second/bin/conda init
```

### 切换和使用不同的 Conda 实例
```bash
# 激活第一个 Conda 实例
source $HOME/miniconda3/bin/activate

# 激活第二个 Conda 实例
source $HOME/miniconda3-second/bin/activate
```

## 删除 Conda 环境

```bash
conda remove --name your_env_name --all
```

其中，`your_env_name` 是想要删除的环境的名称。

- `--name` 选项指定要删除的环境的名称。
- `--all` 选项告诉 Conda 删除环境中的所有包。

例如，如果要删除名为 `myenv` 的环境及其中的所有包，你可以运行以下命令：

```bash
conda remove --name myenv --all
```

这会彻底删除名为 `myenv` 的 Conda 环境及其中的所有内容。



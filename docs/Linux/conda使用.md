### 删除 Conda 环境

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

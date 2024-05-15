在Python中，通常使用`pandas`库来创建和操作DataFrame对象。

## 创建 DataFrame

- **从列表或数组创建**：你可以使用列表或NumPy数组来创建一个DataFrame。

```python
import pandas as pd

data = [
    [1, "Alice", 23],
    [2, "Bob", 27],
    [3, "Charlie", 22]
]

# 创建DataFrame
df = pd.DataFrame(data, columns=["ID", "Name", "Age"])

print(df)
```
Ouput:
<pre>
   ID     Name  Age
0   1    Alice   23
1   2      Bob   27
2   3  Charlie   22
</pre>

- **从字典创建**：你也可以使用字典来创建DataFrame，其中字典的键将成为列名，值应为列表或数组。

```python
data = {
    "ID": [1, 2, 3],
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [23, 27, 22]
}

df = pd.DataFrame(data)

print(df)
```
## 新添加数据
  
```py
df_CD = pd.DataFrame(columns=['Gene', 'Similarity', 'Gene1'])


# 遍历CD_genes列表中的每个基因
for i in CD_genes:
    df = embed.compute_similarities(i, CD_genes)
    df['Gene1'] = i
    print(df)
    print("----------")
    df_CD = pd.concat([df_CD, df], ignore_index=True)

```

1. **从CSV或Excel文件创建**：如果你的数据存储在文件中，`pandas`可以直接读取CSV或Excel文件来创建DataFrame。

```python
# 读取CSV文件
df = pd.read_csv("path/to/your/data.csv")

# 读取Excel文件
df = pd.read_excel("path/to/your/data.xlsx")
```

确保你的文件路径是正确的，并且文件格式与内容符合`pandas`的读取要求。

## 设置索引

你可以在创建DataFrame时指定某一列作为索引：

```python
df = pd.DataFrame(data, index=[row[0] for row in data])
print(df)
```

或者在DataFrame创建后设置：

```python
df.set_index("ID", inplace=True)
print(df)
```

## 其他

### 检查数据类型
- 检查所有列的数据类型
```py
# 假设你已经有一个DataFrame对象，名为df
print(df.dtypes)
```
 - 检查指定列的数据类型
```py
# 假设你想查看名为'column_name'的列的数据类型
column_type = df['column_name'].dtype

print(column_type)
```
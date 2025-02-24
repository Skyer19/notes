## 列表

### 创建列表
```python
# 创建空列表
lst = []

# 创建规定长度的列表（用 0作为默认值)
lst = [0] * n

特定的列表
mixed_list = [1, "hello", 3.14]  # 不同类型的元素
nested_list = [[1, 2], [3, 4]]  # 嵌套列表
```

### 操作列表
```python
# 访问
lst[0]

# 增加元素
lst.append(6)        # 在列表末尾追加元素
lst.insert(2, 100)   # 在索引 2 位置插入 100
lst.extend([7, 8])   # 扩展列表（+= 也可以）

# 删除
lst.pop()        # 删除最后一个元素
lst.pop(1)       # 删除索引 1 处的元素
del lst[2]       # 删除索引 2 处的元素
lst.remove(100)  # 按值删除第一个 100
lst.clear()      # 清空列表

# 查找
print(lst.index(3))   # 查找 3 的索引
print(4 in lst)       # 判断 4 是否在列表中

# 排序
lst.sort()        # 升序排序
lst.sort(reverse=True)  # 降序排序
print(sorted(lst))  # 返回排序后的新列表

#  其他
print(len(lst))   # 获取列表长度
print(lst.count(3))  # 统计 3 出现的次数
lst.reverse()    # 反转列表
```


## 字典（Dictionary）

### 字典创建
```python
dict1 = {"name": "Alice", "age": 25, "city": "New York"}
empty_dict = {}  # 空字典
dict2 = dict(name="Bob", age=30)  # 另一种方式
```

### 字典操作

```python

# 增加 & 修改
dict1["gender"] = "Female"  # 添加新键值对
dict1["age"] = 26  # 修改已有键的值

# 删除元素
dict1.pop("age")  # 删除指定键
del dict1["age"]  # 另一种删除方式
dict1.clear()  # 清空字典

# 访问值
print(dict1["name"])  # 直接访问键的值
print(dict1.get("age", "Not Found"))  # 使用 get() 方法，避免 KeyError

# 遍历字典
for key in dict1:
    print(key, dict1[key])  # 遍历键和值

for key, value in dict1.items():
    print(key, value)  # 使用 .items() 遍历键值对

# 获取键、值、键值对
print(dict1.keys())   # 获取所有键
print(dict1.values())  # 获取所有值
print(dict1.items())   # 获取所有键值对

# 合并字典
dict1.update({"hobby": "reading", "age": 27})  # 使用 update() 更新多个键值对

# 检查键是否存在
print("name" in dict1)  # 检查 "name" 是否在字典中

```

## 集合

```python
# 创建集合
set1 = {1, 2, 3, 4, 5}  # 直接定义
set2 = set([3, 4, 5, 6, 7])  # 使用 set() 创建
empty_set = set()  # 创建空集合（不能用 {}，它表示空字典）

# 增加元素
set1.add(6)  # 添加单个元素
set1.update([7, 8, 9])  # 添加多个元素

# 删除元素
set1.remove(3)  # 删除元素（若不存在会报错）
set1.discard(10)  # 删除元素（若不存在不会报错）
set1.pop()  # 随机删除一个元素
set1.clear()  # 清空集合

# 集合运算
set1 = {1, 2, 3, 4, 5}
set2 = {3, 4, 5, 6, 7}

union_set = set1 | set2  # 并集
intersection_set = set1 & set2  # 交集
difference_set = set1 - set2  # 差集（仅在 set1 中的元素）
symmetric_diff_set = set1 ^ set2  # 对称差集（去掉共同部分）

# 其他操作
print(len(set1))  # 获取集合大小
print(2 in set1)  # 判断元素是否存在
print(set1.issubset(set2))  # 判断 set1 是否是 set2 的子集
print(set1.issuperset(set2))  # 判断 set1 是否包含 set2
print(set1.isdisjoint(set2))  # 判断两个集合是否无交集

# 遍历集合
for item in set1:
    print(item)
```
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
empty_set = set()  # 创建空集合

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

## 元组（Tuple） 
一种 不可变（immutable） 的序列

```py
# 创建元组
t1 = (1, 2, 3)  # 定义元组
t2 = tuple([4, 5, 6])  # 使用 tuple() 将列表转换为元组
t3 = (7,)  # 只有一个元素的元组，必须加逗号，否则只是整数
t4 = ()  # 空元组
t5 = (1, "hello", 3.14)  # 元组可以存储不同类型的元素

# 访问元组元素
t = (10, 20, 30, 40, 50)
print(t[0])  # 获取索引 0 处的元素，输出 10
print(t[-1])  # 获取最后一个元素，输出 50
print(t[1:4])  # 切片获取索引 1 到 3 的元素，输出 (20, 30, 40)

# 遍历元组
for item in t:
    print(item)  # 逐个打印元组元素

for i, value in enumerate(t):  # 使用 enumerate 获取索引和值
    print(f"索引 {i}: {value}")

# 元组不可变，但可包含可变对象
t = (1, 2, 3)
# t[0] = 100  # 会报错，元组中的元素不可更改

t_list = ([1, 2, 3], [4, 5, 6])  # 元组可以包含可变对象（如列表）
t_list[0].append(99)  # 修改元组中的列表
print(t_list)  # 输出 ([1, 2, 3, 99], [4, 5, 6])

# 合并和复制元组
t1 = (1, 2, 3)
t2 = (4, 5, 6)
t3 = t1 + t2  # 合并元组
print(t3)  # 输出 (1, 2, 3, 4, 5, 6)

t4 = t1 * 3  # 复制元组
print(t4)  # 输出 (1, 2, 3, 1, 2, 3, 1, 2, 3)

# 查找元素
t = (1, 2, 3, 4, 5, 3, 6)
print(t.index(3))  # 查找 3 的索引（返回第一个匹配项），输出 2
print(t.count(3))  # 统计 3 出现的次数，输出 2

# 元组解包
t = (10, 20, 30)
a, b, c = t  # 直接解包
print(a, b, c)  # 输出 10 20 30

# 使用 * 进行部分解包
t2 = (1, 2, 3, 4, 5)
a, *middle, c = t2  # 变量 *middle 存储中间部分
print(a, middle, c)  # 输出 1 [2, 3, 4] 5

# 元组比较
print((1, 2, 3) == (1, 2, 3))  # True
print((1, 2, 3) < (1, 2, 4))  # True，按元素逐个比较
print((1, 3) > (1, 2, 5))  # True，因为 3 > 2


```

## 使用 enumerate() 函数

```py

# 定义一个列表
items = ["apple", "banana", "cherry"]

# 遍历列表，同时获取索引和值
for index, value in enumerate(items):
    print(f"索引: {index}, 值: {value}")

# 输出:
# 索引: 0, 值: apple
# 索引: 1, 值: banana
# 索引: 2, 值: cherry


# 在元组上使用 enumerate()
t = ("red", "green", "blue")

# 遍历元组，同时获取索引和值
for index, value in enumerate(t):
    print(f"索引: {index}, 颜色: {value}")

# 指定起始索引
items = ["a", "b", "c"]

# enumerate() 指定 start 参数，使索引从 1 开始
for index, value in enumerate(items, start=1):
    print(f"索引: {index}, 值: {value}")

# 输出:
# 索引: 1, 值: a
# 索引: 2, 值: b
# 索引: 3, 值: c


# 在字典上使用 enumerate()
data = {"name": "Alice", "age": 25, "city": "New York"}

# 遍历字典，同时获取索引、键和值
for index, (key, value) in enumerate(data.items()):
    print(f"索引: {index}, 键: {key}, 值: {value}")

# 输出:
# 索引: 0, 键: name, 值: Alice
# 索引: 1, 键: age, 值: 25
# 索引: 2, 键: city, 值: New York

```

## 迭代器（Iterator）
Python 中的一种对象，可以逐个返回元素，使用 `iter()` 生成，`next()` 进行访问

```py
# 创建一个可迭代对象（列表）
nums = [1, 2, 3, 4]

# 使用 iter() 将其转换为迭代器
iter_nums = iter(nums)

# 使用 next() 获取元素
print(next(iter_nums))  # 输出 1
print(next(iter_nums))  # 输出 2
print(next(iter_nums))  # 输出 3
print(next(iter_nums))  # 输出 4
# print(next(iter_nums))  # 若超出范围，会抛出 StopIteration 异常

# 通过 for 循环遍历迭代器
iter_nums = iter(nums)  # 重新创建迭代器
for num in iter_nums:
    print(num)  # 依次输出 1, 2, 3, 4

# 自定义迭代器（必须实现 __iter__() 和 __next__()）
class MyIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self  # 迭代器对象自身

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration  # 迭代结束
        value = self.current
        self.current += 1
        return value

# 创建迭代器实例
my_iter = MyIterator(1, 5)

# 使用 for 循环遍历
for val in my_iter:
    print(val)  # 输出 1, 2, 3, 4

# 使用 next() 逐步获取
my_iter = MyIterator(10, 13)
print(next(my_iter))  # 输出 10
print(next(my_iter))  # 输出 11
print(next(my_iter))  # 输出 12
# print(next(my_iter))  # 超出范围会抛出 StopIteration 异常

# 使用生成器创建迭代器（更简洁）
def my_generator(start, end):
    while start < end:
        yield start  # 生成一个值
        start += 1

# 生成器创建迭代器
gen = my_generator(1, 5)

# 使用 next() 获取生成器中的值
print(next(gen))  # 输出 1
print(next(gen))  # 输出 2
print(next(gen))  # 输出 3
print(next(gen))  # 输出 4
# print(next(gen))  # 超出范围会抛出 StopIteration 异常

# 通过 for 循环遍历生成器
for val in my_generator(10, 13):
    print(val)  # 输出 10, 11, 12
```
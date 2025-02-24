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


## 字符串

```py

# 字符串基本操作

# 创建字符串
s1 = "Hello, World!"
s2 = 'Python is great'
s3 = """多行字符串
可以这样写"""
s4 = '''也可以用单引号'''

# 访问字符串
print(s1[0])   # 访问索引 0 处的字符，输出 'H'
print(s1[-1])  # 访问最后一个字符，输出 '!'
print(s1[:5])  # 切片，获取前 5 个字符，输出 'Hello'

# 遍历字符串
for char in s1:
    print(char)  # 逐个打印字符串中的字符

# 字符串拼接
s5 = s1 + " " + s2  # 使用 + 连接字符串
print(s5)  # 输出 'Hello, World! Python is great'

# 字符串重复
print("Hello " * 3)  # 输出 'Hello Hello Hello '

# 字符串长度
print(len(s1))  # 输出 13

# 查找和替换
print(s1.find("World"))  # 查找 "World" 的索引位置，输出 7
print(s1.replace("World", "Python"))  # 替换 "World" 为 "Python"

# 字符串大小写转换
print(s1.upper())  # 转换为大写
print(s1.lower())  # 转换为小写
print(s1.capitalize())  # 首字母大写
print(s1.title())  # 每个单词首字母大写
print(s1.swapcase())  # 大小写互换

# 去除空格
s6 = "  hello world  "
print(s6.strip())  # 去掉前后空格
print(s6.lstrip())  # 去掉左侧空格
print(s6.rstrip())  # 去掉右侧空格

# 拆分和合并
s7 = "apple,banana,orange"
print(s7.split(","))  # 以 ',' 分割字符串，返回 ['apple', 'banana', 'orange']

words = ["Python", "is", "fun"]
print(" ".join(words))  # 以空格连接列表中的字符串，输出 'Python is fun'

# 判断字符串是否以特定字符开头或结尾
print(s1.startswith("Hello"))  # 判断是否以 "Hello" 开头，返回 True
print(s1.endswith("!"))  # 判断是否以 "!" 结尾，返回 True

# 判断字符串是否为数字或字母
print("123".isdigit())  # 判断是否全为数字，返回 True
print("abc".isalpha())  # 判断是否全为字母，返回 True
print("abc123".isalnum())  # 判断是否全为字母或数字，返回 True

# 统计字符出现次数
print(s1.count("o"))  # 统计 "o" 在 s1 中出现的次数，输出 2

# 格式化字符串
name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old.")  # 使用 f-string 格式化
print("My name is {} and I am {} years old.".format(name, age))  # 使用 format() 格式化
print("My name is %s and I am %d years old." % (name, age))  # 使用 % 进行格式化

# 反转字符串
print(s1[::-1])  # 反转字符串，输出 '!dlroW ,olleH'

```

## 切片操作
```py
```python
# 字符串、列表、元组的切片操作

# 定义字符串、列表和元组
s = "Hello, World!"
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tup = (10, 20, 30, 40, 50, 60)

# 基本切片（[start:end:step]）
print(s[0:5])   # 输出 'Hello'，索引 0 到 4（不包括 5）
print(lst[2:6]) # 输出 [2, 3, 4, 5]，索引 2 到 5（不包括 6）
print(tup[1:4]) # 输出 (20, 30, 40)，索引 1 到 3（不包括 4）

# 省略 start 或 end
print(s[:5])   # 输出 'Hello'，从索引 0 到 4
print(lst[4:]) # 输出 [4, 5, 6, 7, 8, 9]，从索引 4 到末尾
print(tup[:])  # 输出 (10, 20, 30, 40, 50, 60)，整个元组

# 使用步长（step）
print(s[::2])   # 输出 'Hlo ol!'，每隔 2 个字符取一个
print(lst[1:8:2]) # 输出 [1, 3, 5, 7]，从索引 1 到 7，步长 2
print(tup[::-1]) # 输出 (60, 50, 40, 30, 20, 10)，反转元组

# 负索引切片
print(s[-6:-1])  # 输出 'World'，从倒数第 6 到倒数第 2
print(lst[-4:])  # 输出 [6, 7, 8, 9]，从倒数第 4 到末尾
print(tup[:-3])  # 输出 (10, 20, 30)，从索引 0 到倒数第 3（不包括）

# 反向切片
print(s[::-1])  # 输出 '!dlroW ,olleH'，反转字符串
print(lst[::-2]) # 输出 [9, 7, 5, 3, 1]，步长 -2 逆序获取
print(tup[-1:-4:-1]) # 输出 (60, 50, 40)，倒数第 1 到倒数第 4 逆向切片

# 替换部分列表元素
lst[2:5] = [100, 200, 300]  # 修改索引 2 到 4 的值
print(lst)  # 输出 [0, 1, 100, 200, 300, 5, 6, 7, 8, 9]

# 删除部分列表元素
del lst[3:6]  # 删除索引 3 到 5 的元素
print(lst)  # 输出 [0, 1, 100, 7, 8, 9]
```
```
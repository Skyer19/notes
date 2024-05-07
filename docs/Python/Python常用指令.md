
python怎样判断一个目录中的文件有哪些
  ``` py
    directory = path
    entries = os.listdir(directory)
    # 筛选出所有文件，忽略文件夹
    files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
    
    # 打印文件列表
    print(files)
  ``` 

python怎样判断一个文件存不存在
  ``` py
    if os.path.exists(path):
        print("文件存在")
    else:
        print("文件不存在")
  ```


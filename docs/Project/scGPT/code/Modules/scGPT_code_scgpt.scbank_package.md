
## scgpt.scbank.data module
### .DataTable
DataTable 类是一个用于存储和处理单细胞数据表的数据结构。这个类包含了数据加载状态的检查和数据保存的功能，支持以 JSON 和 Parquet 格式保存数据。这个类在生物信息学领域尤其适用于处理单细胞 RNA 测序（scRNA-seq）数据。

```py
classsc gpt.scbank.data.DataTable(
    name: str, 
    data: Dataset | None = None)
```

Methods:

```py
table.save()
```

```py
import pandas as pd

from scgpt.scbank import DataTable
from datasets import Dataset


df = pd.DataFrame({
    "Gene": ["Gene1", "Gene2", "Gene3"],
    "Expression": [10, 20, 30]
})
dataset = Dataset.from_dict(df) # 需要是Dataset格式


# 创建 DataTable 对象
table = DataTable(name="single_cell_data", data=dataset)

# 保存数据为 JSON 格式
table.save(path = "./single_cell_data.json", format = "json")
```

### .MetaInfo

MetaInfo 类是一个用于管理和操作单细胞数据目录元信息的数据结构。它提供了保存、加载和从路径创建元信息对象的方法。这个类在生物信息学领域，特别是在处理和管理大规模单细胞数据时非常有用。

```py
classscgpt.scbank.data.MetaInfo(on_disk_path: Path | str | None = None, on_disk_format: typing_extensions.Literal[json, parquet] = 'json', main_table_key: str | None = None, gene_vocab_md5: str | None = None, study_ids: List[int] | None = None, cell_ids: List[int] | None = None)
```


## scgpt.scbank.databank module

### .DataBank
DataBank 类是一个用于管理和操作大规模单细胞数据的数据结构。它包含多个研究的数据表，可以进行添加、删除、过滤和同步等操作
```py
classscgpt.scbank.databank.DataBank(meta_info: ~scgpt.scbank.data.MetaInfo | None = None, data_tables: ~typing.Dict[str, ~scgpt.scbank.data.DataTable] = <factory>, gene_vocab: dataclasses.InitVar[GeneVocab] = <property object>, settings: ~scgpt.scbank.setting.Setting = <factory>)
```

## scgpt.scbank.monitor module

N/A

## scgpt.scbank.setting module
Setting 类是一个用于配置 DataBank 对象的配置类。它通过 dataclass 装饰器定义，提供了一些用于控制数据加载和处理的选项，包括是否移除零行、批处理大小和是否立即保存数据。

```py
classscgpt.scbank.setting.Setting(
    remove_zero_rows: bool = True, 
    max_tokenize_batch_size: int = 1000000.0, 
    immediate_save: bool = False
    )

```




### .gene_tokenizer.GeneVocab

```py
class scgpt.tokenizer.gene_tokenizer.GeneVocab(gene_list_or_vocab: List[str] | Vocab, specials: List[str] | None = None, special_first: bool = True, default_token: str | None = '<pad>')
```

- Load the vocabulary from a dictionary.
```py
from_dict(token2idx: Dict[str, int], default_token: str | None = '<pad>')→ Self
```

- Load the vocabulary from a file. The file should be either a pickle or a json file of token to index mapping.

```py
from_file(file_path: Path | str)→ Self
```

- Get the pad token.

```py
propertypad_token: str | None
```

- Save the vocabulary to a json file.

```py
save_json(file_path: Path | str)→ None
```

- Set the default token.

```py
set_default_token(default_token: str)→ None
```


### .get_default_gene_vocab

Get the default gene vocabulary, consisting of gene symbols and ids.

```py
scgpt.tokenizer.gene_tokenizer.get_default_gene_vocab()→ GeneVocab
```

### .pad_batch

Pad a batch of data. Returns a list of Dict[gene_id, count].

```py
scgpt.tokenizer.gene_tokenizer.pad_batch(batch: List[Tuple], max_len: int, vocab: Vocab, pad_token: str = '<pad>', pad_value: int = 0, cls_appended: bool = True, vocab_mod: Vocab | None = None)→ Dict[str, Tensor]
```

### .random_mask_value

Randomly mask a batch of data.

```py
scgpt.tokenizer.gene_tokenizer.random_mask_value(values: Tensor | ndarray, mask_ratio: float = 0.15, mask_value: int = -1, pad_value: int = 0)→ Tensor
```

### .tokenize_and_pad_batch

Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).

```py
scgpt.tokenizer.gene_tokenizer.tokenize_and_pad_batch(data: ndarray, gene_ids: ndarray, max_len: int, vocab: Vocab, pad_token: str, pad_value: int, append_cls: bool = True, include_zero_gene: bool = False, cls_token: str = '<cls>', return_pt: bool = True, mod_type: ndarray | None = None, vocab_mod: Vocab | None = None)→ Dict[str, Tensor]
```


### .tokenize_batch

Tokenize a batch of data. Returns a list of tuple (gene_id, count).

```py
scgpt.tokenizer.gene_tokenizer.tokenize_batch(data: ndarray, gene_ids: ndarray, return_pt: bool = True, append_cls: bool = True, include_zero_gene: bool = False, cls_id: int = '<cls>', mod_type: ndarray | None = None, cls_id_mod_type: int | None = None)→ List[Tuple[Tensor | ndarray]]
```
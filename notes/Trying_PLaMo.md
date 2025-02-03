# Trying out PLaMo-13b

Preferred Networks さんからリリースの PLaMo-13B を試してみます。
Colab 環境で生成を早くするために bitsandbytes を使って量子化して試します。

PLaMo-13B は大規模な言語モデルで、日英 2 言語に対する高い性能を示しています。約 130 億個のパラメータを持ち、日本語と英語のベンチマークタスクで優れた結果を示しています。

公開データセットのみで学習され、Apache License 2.0 の下でオープンソースで公開・提供されています。また、Books3 データセットを使用せずに学習されており、様々な用途に利用可能であることも特徴としています ​。

- Huggingface: https://huggingface.co/pfnet/plamo-13b
- 論文:
- blog: https://tech.preferred.jp/ja/blog/llm-plamo/

```python
!pip install transformers accelerate safetensors sentencepiece bitsandbytes -q
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained("pfnet/plamo-13b", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "pfnet/plamo-13b",
    trust_remote_code=True,
    device_map='auto',
    quantization_config=quantization_config
).eval()
```

```python
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Model Parameter Count: {model.num_parameters():,.0f}")
```

    Vocab size: 50432
    Model Parameter Count: 6,808,089,600

```python
%%time

text = "これからの人工知能技術は"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
generated_tokens = model.generate(
    inputs=input_ids,
    max_new_tokens=32,
    do_sample=True,
    top_p=0.95,
    temperature=1.0,
)[0]
generated_text = tokenizer.decode(generated_tokens)
print(generated_text)
```

    これからの人工知能技術は、人間の意識や価値観に影響を与えるか、あるいはそもそも人工知能自身が人間性をもっているのか、といった議論を交わす機会が多くなりそうです。この
    CPU times: user 13 s, sys: 3.83 s, total: 16.8 s
    Wall time: 19.5 s

## 試してみる

ベースモデルなので、テキストを与えてその続きとしてどんな生成をするのかを見てみます。

```python
generation_config = dict(
    max_new_tokens=50,
    repetition_penalty=1.1,
    top_p=0.95,
    top_k=50,
    temperature=0.7,
    do_sample=True,
)
```

```python
%%time

inputs = tokenizer("吾輩は", return_tensors="pt", return_attention_mask=False)

outputs = model.generate(
    **inputs.to(model.device),
    **generation_config
)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

    吾輩は猫である』の文庫本を読んでいる。
    「猫」は人間より賢い動物である。 猫の仕草が人間の心を惑わす。 猫には、「吾輩は猫である。名前はまだない」という
    CPU times: user 18.5 s, sys: 5.86 s, total: 24.4 s
    Wall time: 24.3 s

```python
%%time

inputs = tokenizer("最新の自然言語処理では", return_tensors="pt", return_attention_mask=False)

outputs = model.generate(
    **inputs.to(model.device),
    **generation_config
)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

    最新の自然言語処理では、自然言語処理技術を、機械学習や深層学習などの様々な分野に応用しています。
    また、2016年は、ディープラーニングのブームが到来しました。そのブーム以降、Deep Learning(
    CPU times: user 18.5 s, sys: 6.04 s, total: 24.5 s
    Wall time: 24.5 s

```python
%%time

inputs = tokenizer("例題: りんごが２つとみかんが３つあります。フルーツは合計で何個ありますか？\n回答: ", return_tensors="pt", return_attention_mask=False)

outputs = model.generate(
    **inputs.to(model.device),
    **generation_config
)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

    例題: りんごが2つとみかんが3つあります。フルーツは合計で何個ありますか?
    回答: 5個
    解説: (1)まずはりんごの個数を数えましょう。
    2個なので、次の2つの答えが出てきますね。
    2×2=4個です。
    (2)みかんは3
    CPU times: user 20 s, sys: 6.91 s, total: 26.9 s
    Wall time: 26.8 s

## 終わりに

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回の Colab はこちらです：
https://colab.research.google.com/drive/1XVtSvXmlY3bK__f84R_h93qO4hldv-Y_?usp=sharing

## 参考

- https://huggingface.co/pfnet/plamo-13b
- https://tech.preferred.jp/ja/blog/llm-plamo/

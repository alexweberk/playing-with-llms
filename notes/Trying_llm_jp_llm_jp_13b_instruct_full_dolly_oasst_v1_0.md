# llm-jp/llm-jp-13b-instruct-full-dolly-oasst-v1.0 を Colab で試す

少し出遅れてしまったが、新たにリリースされた日本語 LLM「llm-jp」を試してみたいと思います。複数バージョンがありますが、「jaster を含むものは回答がそっけない」ということを Twitter で聞いた気がしたので、今回はそれを含まないものを試してみたいと思います。

Huggingface:

- https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-dolly-oasst-v1.0

```python
!pip install transformers accelerate sentencepiece --quiet
```

## モデルのダウンロード

```python
%time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "llm-jp/llm-jp-13b-instruct-full-dolly-oasst-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="cuda:0",
    torch_dtype=torch.float16,
).eval()
```

```python
tokenizer.vocab_size
```

    50570

```python
generation_config = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
}

text = "自然言語処理とは何か"
text = text + "### 回答："

with torch.no_grad():
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        **generation_config,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

    自然言語処理とは何か### 回答：自然言語処理（NLP）は、コンピュータ・プログラムが人間の言語を処理するためのプロセスである。自然言語処理（NLP）は、コンピュータ・プログラムが人間の言語を処理するためのプロセスである。NLPは、コンピュータ・プログラムが人間の言語を処理するためのプロセスである。このプロセスには、テキストの分析、テキストの要約、テキストの生成、テキストの翻訳、テキストの要約などが含まれる。NLPは、音声認識、テキスト要約、翻訳などのアプリケーションに使用されている。

## テンプレートの準備

生成が楽になるようにテンプレートを準備します。

```python
def format_prompt(
    prompt: str,
    system_prompt: str = "" # 今回特に必要ない
) -> str:
    prompt = prompt + "### 回答： "
    prompt = system_prompt + prompt
    return prompt

format_prompt("１＋１は何？")
```

    '１＋１は何？### 回答： '

```python
def ask(
    prompt: str,
    system_prompt: str | None = "",
    **kwargs
) -> str:
    generation_config = {
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    generation_config.update(kwargs)

    with torch.no_grad():
        prompt = format_prompt(prompt, system_prompt)
        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors='pt',
        ).to(model.device)

        # .to(model.device)
        outputs = model.generate(
            **inputs,
            **generation_config,
        )

    output = tokenizer.decode(outputs[0])

    print(output)
    return output

ask("六本木周辺の観光スポットを教えてください。");
```

    六本木周辺の観光スポットを教えてください。### 回答： 六本木ヒルズと東京ミッドタウンは、東京の中心的なランドマークとなっている。<EOD|LLM-jp>

## 質問してみる

```python
text = """
りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？
""".strip()

ask(text);
```

    りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？### 回答： 5-2=3 答えは3つ<EOD|LLM-jp>

```python
text = """バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？""".strip()

ask(prompt=text);
```

    バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？### 回答： ボールは1個100円。バットは1本1100円。<EOD|LLM-jp>

```python
text = """
引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。
""".strip()

ask(prompt=text);
```

    引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。### 回答： ``python def fib(n): if n < 2: return 0 return n + fib(n - 1) ```` ``` >>> fib(5) # フィボナッチ数列の5番目の値は13 ```<EOD|LLM-jp>

```python
text = """
下記の英語を日本語に翻訳してください。
`There were 3 apples and 2 oranges. How many fruits were there in total?`
""".strip()

ask(prompt=text);
```

    下記の英語を日本語に翻訳してください。
    `There were 3 apples and 2 oranges. How many fruits were there in total?`### 回答： 合計で4つの果物があった。<EOD|LLM-jp>

```python
text = """
下記の文章を要約してください。

``

【速報】ロシア月探査機「ルナ 25」が月に衝突 「消滅した」ロスコスモス発表
月に向かっていたロシアの無人探査機「ルナ 25」が月に衝突したことが分かった。ロシアの国営宇宙企業ロスコスモスは先ほど、「月に衝突し、消滅した」と明らかにした。月面着陸前の軌道に移行中、制御不能となったという。
探査機は 21 日に月の南極付近に着陸予定だった。

``
""".strip()

ask(prompt=text);
```

    下記の文章を要約してください。

    ``
    【速報】ロシア月探査機「ルナ25」が月に衝突 「消滅した」ロスコスモス発表
    月に向かっていたロシアの無人探査機「ルナ25」が月に衝突したことが分かった。ロシアの国営宇宙企業ロスコスモスは先ほど、「月に衝突し、消滅した」と明らかにした。月面着陸前の軌道に移行中、制御不能となったという。
    探査機は21日に月の南極付近に着陸予定だった。
    ``### 回答： 月に衝突した。<EOD|LLM-jp>

```python
text = """
あなたは友達ボットです。できるだけユーザーが親近感を感じやすいよう接してください。

ユーザー: 今日バイト、クビにされたわー。あー人生の意味って何なんだろうねぇー。
アシスタント:
""".strip()

ask(prompt=text);
```

    あなたは友達ボットです。できるだけユーザーが親近感を感じやすいよう接してください。

    ユーザー: 今日バイト、クビにされたわー。あー人生の意味って何なんだろうねぇー。
    アシスタント:### 回答： なぜあなたはクビにされたのですか？<EOD|LLM-jp>

```python
text = """
### Question
There was a cookie on the table.
Tom entered the room.
The cookie disappeared.
What was likely to have happened?
""".strip()

ask(prompt=text);
```

    ### Question
    There was a cookie on the table.
    Tom entered the room.
    The cookie disappeared.
    What was likely to have happened?### 回答： トムがテーブルの上のクッキーを食べた。<EOD|LLM-jp>

```python
text = """
### 質問
テーブルにクッキーがおいてありました。
太郎が部屋に入りました。
クッキーが消えました。
何が起きた可能性が高いですか？
""".strip()

ask(prompt=text);
```

    ### 質問
    テーブルにクッキーがおいてありました。
    太郎が部屋に入りました。
    クッキーが消えました。
    何が起きた可能性が高いですか？### 回答： クッキーが太郎の部屋から消えたということは、太郎がクッキーを食べたことを意味する。太郎がクッキーを食べたなら、太郎がクッキーを太郎の部屋に置いたことになる。<EOD|LLM-jp>

```python
text = """
必ず関西弁で答えてください。
たこ焼きのレシピを教えてください。
""".strip()

ask(prompt=text);
```

    必ず関西弁で答えてください。
    たこ焼きのレシピを教えてください。### 回答： もちろん！材料と作り方は以下の通り：

      * タコ
      * 小麦粉
      * タコソース（お好みで）
      * 天かす
      * ソース
      * マヨネーズ
      * 青のり

     ステップ1：タコを切る。タコは1cm幅に切る。
    ステップ2：小麦粉をボールに入れ、水を少しずつ加えて混ぜる。
    ステップ3：タコを小麦粉に加え、タコと小麦粉がなじむまで混ぜる。
    ステップ4：小麦粉に天かすを加え、さらに混ぜる。

## まとめ

流石日本語特化のモデルだけあって日本語は自然な形で生成できました。日本に関する基本的な知識も備えているのは嬉しいですね。

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回の Colab はこちらです：
https://colab.research.google.com/drive/139GacVjsU1OBZ2pFXf9gKOhOarH4Te4d?usp=sharing

## 参考

- https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-dolly-oasst-v1.0

#機械学習 #AI #自然言語処理 #python #LLM

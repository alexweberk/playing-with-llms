# HuggingFaceH4/zephyr-7b-alpha を試す

今回試してみる Zephyr-7B-alpha は、Hugging Face 社によって開発された言語モデルのシリーズで、 mistral-7b に対するファインチューンとなっています。
ChatGPT で生成した会話系の合成データセットの UltraChat を使っており、Direct Preference Optimization (DPO) という手法を使ってトレーニングされています。
DPO についてはまだ完全に理解できていませんが、RLHF (Reinforcement Learning from Human Feedback) の代わりとなる方法と理解しています。

Huggingface: https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha

## Colab で試す

```python
!pip install transformers accelerate sentencepiece bitsandbytes -Uqq
```

## モデルのダウンロード

モデルカードでは `pipeline` を使っていますが、量子化の方法が分からなかったので`AutoModelForCausalLM` を使うよう書き換えました。

```python
%time

import torch
from transformers import pipeline, BitsAndBytesConfig

model_name = "HuggingFaceH4/zephyr-7b-alpha"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto",
).eval()
```

```python
tokenizer.vocab_size
```

    32000

```python
generation_config = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
}

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]

with torch.no_grad():
    inputs = tokenizer(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), # このモデル用のチャットテンプレートをテキストとして生成
        return_tensors='pt',
    ).to(model.device)

    # .to(model.device)
    outputs = model.generate(
        **inputs,
        **generation_config,
        use_cache=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

    <|system|>
    You are a friendly chatbot who always responds in the style of a pirate
    <|user|>
    How many helicopters can a human eat in one sitting?
    <|assistant|>
    Matey, I be sayin' that a human cannot eat a helicopter, as it is not food. Ye be confused, me hearty.

無事プロンプト通りの海賊風の文章が生成できました。

## テンプレートの準備

生成が楽になるようにテンプレートを準備します。

```python
def format_prompt(
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    if system_prompt is None:
        system_prompt = "You are a friendly chatbot. You are extremely detailed and factual. You are honest when you don't have an answer."

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

# tokenizer.apply_chat_templateの中身を見てみる
format_prompt("こんにちは")
```

    "<|system|>\nYou are a friendly chatbot. You are extremely detailed and factual. You are honest when you don't have an answer.</s>\n<|user|>\nこんにちは</s>\n"

```python
def ask(
    prompt: str,
    system_prompt: str | None = None,
    **kwargs
) -> str:
    generation_config = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 0.95,
    }
    generation_config.update(kwargs)

    with torch.no_grad():
        prompt = format_prompt(prompt, system_prompt)
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
        ).to(model.device)

        # .to(model.device)
        outputs = model.generate(
            **inputs,
            **generation_config,
            use_cache=True,
        )

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(output)
    return output

ask("How did the first President of the United States become a President?");
```

    <|system|>
    You are a friendly chatbot. You are extremely detailed and factual. You are honest when you don't have an answer.
    <|user|>
    How did the first President of the United States become a President?
    <|assistant|>
    The first President of the United States, George Washington, became President through the process of election. In 1788, the United States Constitution was ratified, and the first presidential election was held in 1789. The Electoral College, a group of electors appointed by the people in each state, cast their votes for the President and Vice President. Washington received the majority of electoral votes and was inaugurated as the first President of the United States on April 30, 1789.

## 質問してみる

```python
system_prompt_ja = "あなたはトップクラスのアシスタントです。必ず日本語で答えてください。"
```

```python
text = """
りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？
""".strip()

ask(prompt=text, system_prompt=system_prompt_ja);
```

    <|system|>
    あなたはトップクラスのアシスタントです。必ず日本語で答えてください。
    <|user|>
    りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？
    <|assistant|>
    残りのりんごの数は3個です。

```python
text = """バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？""".strip()

ask(prompt=text, system_prompt=system_prompt_ja);
```

    <|system|>
    あなたはトップクラスのアシスタントです。必ず日本語で答えてください。
    <|user|>
    バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？
    <|assistant|>
    ボールの価格は、バットの価格からバットの価格からバットの価格を引いたものです。

    バットの価格は1000円です。

    バットの価格からバットの価格を引いたものは、1000円から1000円を引いたものです。

    その結果、ボールの価格は、1000円から1000円を引いたものです。

    1000円から1000円を引いたものは、0円です。

    したがって、ボールの価格は0円です。

    しかし、この価格は不正な価格です。

    ボールの

```python
text = """バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？""".strip()

ask(prompt=text, system_prompt=system_prompt_ja + "深呼吸をしたあとに一歩ずつ考えましょう。");
```

    <|system|>
    あなたはトップクラスのアシスタントです。必ず日本語で答えてください。深呼吸をしたあとに一歩ずつ考えましょう。
    <|user|>
    バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？
    <|assistant|>
    ボールの価格は、バットの価格からバットの価格からバットの価格を引いたものになります。

    バットの価格は1000円です。

    バットの価格からバットの価格を引いたものは、1000円から1000円を引いたものになります。

    その結果、ボールの価格は、1000円から1000円を引いたものになります。

    1000円から1000円を引いたものは、0円です。

    したがって、ボールの価格は0円です。

    しかし、この価格は不正な価格

```python
text = """バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？""".strip()

ask(prompt=text, system_prompt="You are an expert Mathematician. Think and answer in English. Then translate the final result in Japanese to provide the final answer.");
```

    <|system|>
    You are an expert Mathematician. Think and answer in English. Then translate the final result in Japanese to provide the final answer.
    <|user|>
    バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？
    <|assistant|>
    Mathematically, if the total cost of buying both a bat and a ball is 1100 yen, and the bat costs 1000 yen more than the ball, then the cost of the ball is:

    Let x be the cost of the ball.
    Let y be the cost of the bat.

    y = x + 1000

    Total cost = x + 1000 + x

    1100 = 2x + 1000

    2x = 1100 - 1000

    2x = 100

    x = 50

    So, the cost of the ball is 50 yen.

    In Japanese:

    バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールは50円です。

```python
text = """
引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。
""".strip()

ask(prompt=text, system_prompt=system_prompt_ja);
```

    <|system|>
    あなたはトップクラスのアシスタントです。必ず日本語で答えてください。
    <|user|>
    引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。
    <|assistant|>
    ```python
    def fibonacci(k):
        if k <= 1:
            return k
        else:
            a, b = 0, 1
            for i in range(k-1):
                a, b = b, a + b
            return b
    ```

    この関数は、引数kを取り、k個目のフィボナッチ数列の値を返します。

    この関数は、kが1以下の場合、その値を返します。そうでない場合、aとbを初期化し、aとbを交換していき、k-1回ループを回し、最終的にbを返します。

    この関数は、フィボナッチ数列の定義に従しています。

```python
text = """
下記の英語を日本語に翻訳してください。
English: There were 3 apples and 2 oranges. How many fruits were there in total?
""".strip()

ask(prompt=text, system_prompt=system_prompt_ja);
```

    <|system|>
    あなたはトップクラスのアシスタントです。必ず日本語で答えてください。
    <|user|>
    下記の英語を日本語に翻訳してください。
    English: There were 3 apples and 2 oranges. How many fruits were there in total?
    <|assistant|>
    日本語: 合計、3つのアップルと2つのオレンジがありました。そのため、合計の果物は5つです。

```python
text = """
There were 3 apples and 2 oranges. How many fruits were there in total?
""".strip()

ask(prompt=text, system_prompt="下記の英語を日本語に翻訳してください。");
```

    <|system|>
    下記の英語を日本語に翻訳してください。
    <|user|>
    There were 3 apples and 2 oranges. How many fruits were there in total?
    <|assistant|>
    合計は3つのアップルと2つのオレンジで6つの果物があります。

```python
text = """
下記の文章をJSON形式に変換してください。

``
【速報】ロシア月探査機「ルナ 25」が月に衝突 「消滅した」ロスコスモス発表
月に向かっていたロシアの無人探査機「ルナ 25」が月に衝突したことが分かった。ロシアの国営宇宙企業ロスコスモスは先ほど、「月に衝突し、消滅した」と明らかにした。月面着陸前の軌道に移行中、制御不能となったという。
探査機は 21 日に月の南極付近に着陸予定だった。
``
""".strip()

ask(prompt=text, system_prompt=system_prompt_ja);
```

    <|system|>
    あなたはトップクラスのアシスタントです。必ず日本語で答えてください。
    <|user|>
    下記の文章をJSON形式に変換してください。

    ```
    【速報】ロシア月探査機「ルナ25」が月に衝突 「消滅した」ロスコスモス発表
    月に向かっていたロシアの無人探査機「ルナ25」が月に衝突したことが分かった。ロシアの国営宇宙企業ロスコスモスは先ほど、「月に衝突し、消滅した」と明らかにした。月面着陸前の軌道に移行中、制御不能となったという。
    探査機は21日に月の南極付近に着陸予定だった。
    ```
    <|assistant|>
    ```
    {
      "title": "【速報】ロシア月探査機「ルナ25」が月に衝突",
      "content": "月に向かっていたロシアの無人探査機「ルナ25」が月に衝突したことが分かった。ロスコスモスは先ほど、「月に衝突し、消滅した」と明らかにした。月面着陸前の軌道に移行中、制御不能となったという。探査機は21日に月の南極付近に着陸予定だった。"
    }

```python
text = """
あなたは友達ボットです。できるだけユーザーが親近感を感じやすいよう接してください。

ユーザー: 今日バイト、クビにされたわー。あー人生の意味って何なんだろうねぇー。
アシスタント:
""".strip()

ask(prompt=text, system_prompt=system_prompt_ja);
```

    <|system|>
    あなたはトップクラスのアシスタントです。必ず日本語で答えてください。
    <|user|>
    あなたは友達ボットです。できるだけユーザーが親近感を感じやすいよう接してください。

    ユーザー: 今日バイト、クビにされたわー。あー人生の意味って何なんだろうねぇー。
    アシスタント:
    <|assistant|>
    アシスタント: そうです。人生の意味は、個人によって異なります。たとえば、一部の人が、人生の目的として、家族を育て、社会に貢献したいと考えるかもしれません。その他には、役に立てることや、自己の成長や成長のために努力したいと考えるかもしれません。しかし、人生の意味は、個人によって異なります。私は、あなたの目的や理想を聞き、それを実現するために協力していきたいと考えています。

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

    <|system|>
    You are a friendly chatbot. You are extremely detailed and factual. You are honest when you don't have an answer.
    <|user|>
    ### Question
    There was a cookie on the table.
    Tom entered the room.
    The cookie disappeared.
    What was likely to have happened?
    <|assistant|>
    Tom likely ate the cookie.

```python
text = """
### 質問
テーブルにクッキーがおいてありました。
太郎が部屋に入りました。
クッキーが消えました。
何が起きた可能性が高いですか？
""".strip()

ask(prompt=text, system_prompt=system_prompt_ja);
```

    <|system|>
    あなたはトップクラスのアシスタントです。必ず日本語で答えてください。
    <|user|>
    ### 質問
    テーブルにクッキーがおいてありました。
    太郎が部屋に入りました。
    クッキーが消えました。
    何が起きた可能性が高いですか？
    <|assistant|>
    太郎がクッキーを食べてしまった可能性が高いです。

```python
text = """
たこ焼きのレシピを教えてください。
""".strip()

ask(prompt=text, system_prompt="必ず関西弁で答えてください。", repetition_penalty=1.1);
```

    <|system|>
    必ず関西弁で答えてください。
    <|user|>
    たこ焼きのレシピを教えてください。
    <|assistant|>
    おします。関西弁では、たこ焼きのレシピとして「たこ焼き」と言うものがあります。それは、たこ肉に入ったタコ焼き料理です。以下のレシップをご紹介します。

    材料：
    - 10個のタコ
    - 2つのトマト
    - 1本のネギ
    - 1本のアジテル
    - 1/4カップのソイソース
    - 1/4カップのオイル
    - 1/4カップの醤油
    - 1/4カップのウイン
    - 1/4カップのバラミン
    - 1/4カップのグラニール
    - 1/4カップのブランドワイン
    - 1/4カップのサスエー
    - 1/4カップ

## まとめ

今回は HuggingFaceH4/zephyr-7b-alpha を試してみました。日本語のトレーニングが無いにも関わらず、それなりに日本語の質問にも答えられましたが、日本人の求める日本文化への理解などはなく、単語の使い方などについても限界はありました。

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Colab:
https://colab.research.google.com/drive/1ia61dBinuAtReQd5mCVRL-bOxJDLkwNj?usp=sharing

## 参考

- https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
- Direct Preference Optimization (DPO): https://arxiv.org/abs/2305.18290

## 関連

https://note.com/alexweberk/n/n155425dcd0a8
https://note.com/alexweberk/n/n59d0de2d7a76
https://note.com/alexweberk/n/n439fc8264668

# Trying Swallow 13B Instruct HF

Llama 2 ベースの新しい日本語特化言語モデル Swallow 13B Instruct HF を Colab で試しました。

- モデル: https://huggingface.co/tokyotech-llm/Swallow-13b-instruct-hf

## セットアップ

```python
!pip install accelerate sentencepiece -Uqq
```

```python
import torch
print(torch.version)
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
```

    <module 'torch.version' from '/usr/local/lib/python3.10/dist-packages/torch/version.py'>
    2.1.0+cu121
    12.1
    8902
    True

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "tokyotech-llm/Swallow-13b-instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto")
```

```python
tokenizer.vocab_size
```

    43176

```python

PROMPT_DICT = {
    "prompt_input": (
        "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。"
        "リクエストを適切に完了するための回答を記述してください。\n\n"
        "### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:"

    ),
    "prompt_no_input": (
        "以下に、あるタスクを説明する指示があります。"
        "リクエストを適切に完了するための回答を記述してください。\n\n"
        "### 指示:\n{instruction}\n\n### 応答:"
    ),
}

def create_prompt(instruction, input=None):
    """
    Generates a prompt based on the given instruction and an optional input.
    If input is provided, it uses the 'prompt_input' template from PROMPT_DICT.
    If no input is provided, it uses the 'prompt_no_input' template.

    Args:
        instruction (str): The instruction describing the task.
        input (str, optional): Additional input providing context for the task. Default is None.

    Returns:
        str: The generated prompt.
    """
    if input:
        # Use the 'prompt_input' template when additional input is provided
        return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
    else:
        # Use the 'prompt_no_input' template when no additional input is provided
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)

# Example usage
instruction_example = "以下のトピックに関する詳細な情報を提供してください。"
input_example = "東京工業大学の主なキャンパスについて教えてください"
prompt = create_prompt(instruction_example, input_example)

input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
)

tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=128,
    temperature=0.99,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)

```

    以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。リクエストを適切に完了するための回答を記述してください。

    ### 指示:
    以下のトピックに関する詳細な情報を提供してください。

    ### 入力:
    東京工業大学の主なキャンパスについて教えてください

    ### 応答:東京工業大学大岡山キャンパスは、東京都目黒区に位置し、大岡山駅(東急大井町線)から歩いて5分程度の場所にあります。1929年に、日本の最初の国立工科大学として開学した東京工業大学の本部キャンパスでもあります。

質問しやすいよう関数を作ります。

```python
from transformers import TextStreamer

def ask(prompt: str, system_prompt: str | None = None, return_output: bool = False) -> str:
    if system_prompt is None:
        system_prompt = prompt
        prompt = None
    prompt = create_prompt(system_prompt, prompt)

    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids.to(device=model.device),
            max_new_tokens=128,
            temperature=0.99,
            top_p=0.95,
            do_sample=True,
            streamer=TextStreamer(tokenizer, skip_prompt=True)
        )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if return_output:
        return out

out = ask("日本で一番高い山はなんですか？", return_output=True)
out
```

    富士山が日本で一番高い山です。

```python
ask("Give me a short answer. What are the first 10 numbers in the fibonacci sequence?")
```

    1, 1, 2, 3, 5, 8, 13, 21, 34, 55

```python
ask("How did the first President of the United States become a President?")
```

    第1代大統領とは、アメリカの初代大統領、ジョージ・ワシントンのことです。

```python
ask("りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？")
```

    残りのりんごの数は3つです。

```python
ask("バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？")
```

    バットは1000円高いので、バットは1100 + 1000 = 2100円でなければなりません。バットより1000円安いボールはいくらなのか?

    1100 - 1000 = 100なので、ボールは100円です。

```python
ask("A bat and a ball costs $11 total. The bat is $10 more than the ball. How much is the ball")
```

    バットはボールより10ドル高い。ボールの価格は1ドル。バットの価格は11ドル。11 + 1 = 12。ボールの価格は1ドル。batの価格は12 - 1 = 11。バットはボールより10ドル高い。12 - 10 = 2。バットの価格は2ドル。ボールの価格は1ドル。よって、ボールは2ドルとなる。

```python
ask("引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。")
```

    フィボナッチ数列を求める関数は以下のとおりです。
    ```python
    def fibonacci(n):
       if n == 0 or n == 1:
           return n

       return fibonacci(n - 1) + fibonacci(n - 2)
    ```

```python
ask("""下記の英語を日本語に翻訳してください。
English: There were 3 apples and 2 oranges. How many fruits were there in total?""")
```

    りんごは3個、オレンジは2個。全部で5個のフルーツがありました。

```python
ask("""
あなたは友達ボットです。できるだけユーザーが親近感を感じやすいよう接してください。

ユーザー: 今日バイト、クビにされたわー。あー人生の意味って何なんだろうねぇー。
アシスタント:
""")
```

    今日クビになったの?私がお金を稼いであげたらいいのに、私はたくさんお金を稼いでいるの!
    今バイトが必要ではない?私はとてもよく働くよ!
    どうして人生の意味がわからないの?私はすでにそれを知っているよ!

```python
ask("""
### Question
There was a cookie on the table.
Tom entered the room.
The cookie disappeared.
What was likely to have happened?
""")
```

    彼は食べた。

```python
ask("""
### 質問
テーブルにクッキーがおいてありました。
太郎が部屋に入りました。
クッキーが消えました。
何が起きた可能性が高いですか？
""")
```

    太郎がクッキーを食べた

```python
ask("たこ焼きのレシピを教えてください。")
```

    小麦粉と卵と水を混ぜた生地と、キャベツの葉、紅しょうが、タコを入れて混ぜ、油を引いた鉄板に生地を入れ、タコの上に入れます。

```python
ask("大規模言語モデルについて説明してください。")
```

    大規模な言語モデルは、人間のようなものを書くことを目的としたコンピューターシステムです。

```python
ask("""間違いがないよう、正確に質問に答えてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。夏菜子には何人の子供がいますか？
""")
```

    1人

```python
ask("""下記の文章における登場人物とその関係性をJSON形式でアウトプットしてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。
""")
```

    太郎 :兄
    次郎 :弟
    夏菜子 :次郎の母
    次郎 :２４歳

```python
ask("""下記の文章をUTF-8形式のJSON形式でアウトプットしてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。
""")
```

    「"太郎は二郎の母親です。": True "二郎は２４歳です。": True "兄妹": True "夏菜子": True "年上": False "母親": False "太郎": True "二郎": False "年齢": False "兄妹": False "母親": True "年齢": False "太郎": True "二郎": True "夏菜子": False "年上": True "母親": False "兄妹": False "年齢": False "太郎": True "二郎": False "夏菜子": False "年上": False "母親

```python
ask("""下記の文章をJSON形式でアウトプットしてください。必ず下記のフォーマットで、全ての登場人物に関して出力してください。

## フォーマット
{
    "name": "<<名前>>",
    "age": <<年齢>>,
    "family": ["<<家族の名前>>", ...]
}

## 文章
太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。
""")
```

    {
     "name": "太郎",
     "age": "25",
     "family": [
       "夏菜子",
       "二郎",
     ]
    }
    {
     "name": "二郎",
     "age": "20",
     "family": [
       "夏菜子",
     ]
    }

## おわりに

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Colab:
https://colab.research.google.com/drive/1yD15x9f0qiAmuFzHkQk5rDvlFAChnGWO?usp=sharing

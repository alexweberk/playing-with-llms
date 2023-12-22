# Trying openchat/openchat-3.5-1210

7B パラメータ級でトップクラス（？）のベンチマークを握る OpenChat3.5-1210 を Colab で試してみました。
意外に日本語が使えてびっくりです。
体感でも GPT-3.5 くらいの品質があり、利便性のあるモデルだと思いました。

- モデル: https://huggingface.co/openchat/openchat-3.5-1210

```python
!pip install accelerate sentencepiece -Uqq
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_name = "openchat/openchat-3.5-1210"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
```

```python
tokenizer.vocab_size
```

    32000

```python
# サンプルにあったコードが走るか
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
    {"role": "user", "content": "How are you today?"}
]
tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747, 15359, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 1602, 460, 368, 3154, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]

```

```python
# 注意書きがあったので
tokenizer.eos_token = '<|end_of_turn|>'
```

```python
# まずは英語で適当な質問を
prompt = "The ball and bat are $11 total. The bat is $10 more than the ball. How much is the ball?"
messages = [
    {"role": "user", "content": "You are an extremely helpful assistant. You are extremely detail oriented. Answer user's questions by first reiterating the question and then thinking step by step."},
    {"role": "user", "content": prompt}
]
tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
with torch.no_grad():
    out = model.generate(
        tokens.to(model.device),
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        do_sample=True,
        streamer=TextStreamer(tokenizer, skip_prompt=True)
    )
out
```

    Let's break this down step by step:

    1. We know that the total cost of the ball and bat is $11.
    2. We also know that the bat is $10 more expensive than the ball.

    Let's use variables to represent the unknowns:
    Let B = the cost of the ball.
    Let B+10 = the cost of the bat.

    Now we can set up two equations based on the information given:

    Equation 1: B + (B + 10) = $11 (the total cost of the ball and bat)
    Equation 2: B + 10 = B + $10 (the bat is $10 more expensive than the ball)

    Now we can solve the equations:

    From Equation 2, we can subtract B from both sides:
    10 = $10

    This equation is true, which means that the information given is inconsistent or there is not enough information to solve the problem. Please check the information provided and try again.<|end_of_turn|>

```python
def ask(prompt: str, system_prompt: str = "", return_output: bool = False) -> str:
    messages = []
    if system_prompt != "":
        messages.append({"role": "user", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')

    with torch.no_grad():
        out = model.generate(
            tokens.to(model.device),
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        )

    if return_output:
        out = tokenizer.decode(out[0], skip_special_tokens=True)
        return out

out = ask("日本で一番高い山はなんですか？", return_output=True)
out
```

    日本で一番高い山は、富士山（Fujiyama）です。高さは3776メートルです。<|end_of_turn|>

```python
out = ask("日本で一番高い山はなんですか？", return_output=True)
out
```

    日本で一番高い山は、富士山（Fujiyama）です。高さは3776メートルです。<|end_of_turn|>

```python
ask("Give me a short answer. What are the first 10 numbers in the fibonacci sequence?")
```

    The first 10 numbers in the Fibonacci sequence are: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34.<|end_of_turn|>

```python
ask("How did the first President of the United States become a President?")
```

    George Washington became the first President of the United States through a combination of his military leadership during the American Revolutionary War and his political influence in the early years of the new nation.

    Washington was a prominent figure in the American colonies even before the Revolutionary War began. He was a successful planter and landowner in Virginia, and he had a reputation for his strong leadership and military skills. When the war broke out in 1775, Washington was appointed as the commander-in-chief of the Continental Army by the Second Continental Congress.

    Under Washington's leadership, the Continental Army managed to defeat the British forces, securing American independence. His military success and popularity made him a natural choice for the new nation's first president when the United States Constitution was ratified in 1789.

    Washington was unanimously elected as the first President of the United States by the Electoral College, and he took office on April 30, 1789. He served two terms, setting a precedent for future presidents to serve only two terms. Washington's leadership during the Revolutionary War and his successful presidency established him as a key figure in American history.<|end_of_turn|>

```python
ask("りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？")
```

    残りのりんごの数は3個です。<|end_of_turn|>

```python
ask("バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？")
```

    バットとボールの合計価格が1100円で、バットがボールよりも1000円高いという情報があります。これらの情報から、バットの価格は1100円 - ボールの価格 = 1000円です。

    それでは、ボールの価格は1100円 - 1000円 = 100円です。<|end_of_turn|>

```python
ask("A bat and a ball costs $11 total. The bat is $10 more than the ball. How much is the ball")
```

    Let's denote the cost of the ball as B and the cost of the bat as B+$10 (since the bat is $10 more than the ball). The problem states that the total cost of the bat and the ball is $11. Therefore, we can set up the following equation:

    B + (B + $10) = $11

    Combining like terms, we get:

    2B + $10 = $11

    Subtracting $10 from both sides, we get:

    2B = $1

    Dividing both sides by 2, we find:

    B = $0.50

    So, the ball costs $0.50.<|end_of_turn|>

```python
ask("引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。")
```

    以下は、引数kを取り、フィボナッチ数列におけるk個目の値を返すPython関数の例です。

    ```python
    def fibonacci(k):
       if k <= 0:
           return "kは正の整数である必要があります。"
       elif k == 1:
           return 0
       elif k == 2:
           return 1
       else:
           a, b = 0, 1
           for i in range(2, k):
               a, b = b, a + b
           return b

    # 使用例
    k = 10
    print(f"フィボナッチ数列の{k}番目の値は{fibonacci(k)}です。")
    ```

    この関数は、kが1の場合は0を返し、kが2の場合は1を返します。kが3以上の場合は、フィボナッチ数列を生成してk番目の値を返します。<|end_of_turn|>

```python
ask("""下記の英語を日本語に翻訳してください。
English: There were 3 apples and 2 oranges. How many fruits were there in total?""")
```

    合計で何個の果物がありましたか？<|end_of_turn|>

```python
ask("""
あなたは友達ボットです。できるだけユーザーが親近感を感じやすいよう接してください。

ユーザー: 今日バイト、クビにされたわー。あー人生の意味って何なんだろうねぇー。
アシスタント:
""")
```

    すみません、それは本当に困っていることを感じました。でも、人生の意味は異なる人々にとって異なるものがあります。重要なのは、自分が楽しいことや意味のあることを見つけることですね。何かお手伝いできることがあれば、遠慮なくお聞かせくださいね。<|end_of_turn|>

```python
ask("""
### Question
There was a cookie on the table.
Tom entered the room.
The cookie disappeared.
What was likely to have happened?
""")
```

    Tom likely ate the cookie.<|end_of_turn|>

```python
ask("""
### 質問
テーブルにクッキーがおいてありました。
太郎が部屋に入りました。
クッキーが消えました。
何が起きた可能性が高いですか？
""")
```

    太郎がクッキーを食べた可能性が高いです。<|end_of_turn|>

```python
ask("たこ焼きのレシピを教えてください。")
```

    たこ焼きのレシピを教えてください。

    材料：

    * たこ（3個）
    * 白米（1杯）
    * 卵（2個）
    * 塩（少量）
    * 醤油（少量）
    * 砂糖（少量）

    作り方：

    1. 白米を沸かし、煮き上げて煮き終わるまで煮る。煮き終わったら汁を取り除く。
    2. 卵を蓋をかけて軽く焼き上げる。焼き終わったらたこの形に叩き平らくする。
    3. たこを切り締める。
    4. 醤油、塩、砂糖を混ぜ、たこに抹露する。
    5. 焼き上げた卵をたこの上に詰める。
    6. 焼き焼く。焼き終わったら、塩をかける。

    このレシピでは、たこ焼きを簡単に作ることができます。たこ焼きは、日本の一つのお好み焼きで、家庭で楽しめるお食事です。<|end_of_turn|>

```python
ask("大規模言語モデルについて説明してください。")
```

    大規模言語モデル（Large-scale Language Models）は、自然言語処理（NLP）や人工知能（AI）の分野で重要な技術として、コンピュータによって学習される言語モデルの一種です。これらのモデルは、大量のテキストデータを使ってトレーニングされ、その後、入力された文章から次の単語やフレーズを予測することができます。

    大規模言語モデルは、深層学習（Deep Learning）に基づいています。特に、リンク層（Recurrent Neural Networks, RNN）や自己注意メカニズム（Self-Attention Mechanisms）を用いることが一般的です。これらの技術を用いて、モデルは文章の構造や文脈を理解し、より正確な予測を行うことができます。

    最近の大規模言語モデルは、GoogleのBERTやOpenAIのGPT-3など、さまざまな企業や研究機関が開発しています。これらのモデルは、さまざまな言語処理タスクに適用されることができ、例えば文章の生成、翻訳、傾向分析、情報抽出などに使われています。

    ただし、大規模言語モデルにはいくつかの問題があります。例えば、バイアスや偏見がモデルに入っている可能性があり、不適切な出力を生成することがあります。また、モデルのサイズが大きくなると、計算コストやエネ

```python
ask("""間違いがないよう、正確に質問に答えてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。夏菜子には何人の子供がいますか？
""")
```

    太郎と二郎は兄妹で、二郎は２４歳です。太郎は二郎より５つ年上であることから、太郎は２９歳です。この情報から、夏菜子には二郎が子供であるため、１人の子供がいます。<|end_of_turn|>

```python
ask("""下記の文章における登場人物とその関係性をJSON形式でアウトプットしてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。
""")
```

    {
     "characters": [
       {
         "name": "太郎",
         "relation": "兄",
         "age_difference": 5,
         "sibling": "二郎"
       },
       {
         "name": "二郎",
         "relation": "弟",
         "age": 24,
         "mother": "夏菜子",
         "sibling": "太郎"
       },
       {
         "name": "夏菜子",
         "relation": "母親",
         "child": "二郎"
       }
     ],
     "relationships": [
       {
         "from": "太郎",
         "to": "二郎",
         "relation": "兄妹"
       },
       {
         "from": "二郎",
         "to": "太郎",
         "relation": "兄妹"
       },
       {
         "from": "二郎",
         "to": "夏菜子",
         "relation": "母親"
       }
     ]
    }<|end_of_turn|>

```python
ask("""下記の文章をUTF-8形式のJSON形式でアウトプットしてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。
""")
```

    {
     "Tarou": {
       "age": 29,
       "relationship": "older brother"
     },
     "Ippou": {
       "age": 24,
       "relationship": "younger brother",
       "mother": "Natsumon"
     },
     "Natsumon": {
       "relationship": "mother"
     }
    }<|end_of_turn|>

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
       "Tarou": {
           "age": 29,
           "family": ["Ichiro", "Natsumi", "Summer"]
       },
       "Jiro": {
           "age": 24,
           "family": ["Summer", "Natsumi"]
       },
       "Summer": {
           "age": 45,
           "family": ["Tarou", "Jiro", "Ichiro", "Natsumi"]
       },
       "Ichiro": {
           "age": 70,
           "family": ["Tarou", "Natsumi"]
       },
       "Natsumi": {
           "age": 48,
           "family": ["Tarou", "Jiro", "Ichiro"]
       }
    }<|end_of_turn|>

## おわりに

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Colab:
https://colab.research.google.com/drive/1jFmvSQ6yKqYohBbMOwFP4nJEQa8DQXyL?usp=sharing

# augmxnt/shisa-7b-v1

Mistral 7B をベースに、日本語生成に向くよう対応させたトークナイザーを用い、その後に airoboros-3.1 や ultrafeedback_binarized などを機械翻訳したデータセットに DPO も用いてファインチューニングさせたモデルということで、なかなか楽しみなことをしているモデルです。

- モデル: https://huggingface.co/augmxnt/shisa-7b-v1

同時に、トレーニングに使った日本語データセットも複数同時に公開されており注目です。

- https://huggingface.co/datasets/augmxnt/shisa-pretrain-en-ja-v1
- https://huggingface.co/datasets/augmxnt/ultra-orca-boros-en-ja-v1
- https://huggingface.co/datasets/augmxnt/shisa-en-ja-dpo-v1

```python
!pip install accelerate bitsandbytes -Uqq
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
    2.1.0+cu118
    11.8
    8700
    True

```python
# Wrap printed text
from IPython.display import HTML, display

def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    "augmxnt/shisa-7b-v1",
    use_fast=True
)
model = AutoModelForCausalLM.from_pretrained(
    "augmxnt/shisa-7b-v1",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)
```

```python
# プロンプトの準備
def ask(prompt: str, system_prompt: str = None, return_output: bool = False,) -> str | None:
    chat = []
    if system_prompt is not None:
        chat.append({"role": "system", "content": prompt})

    chat.append({"role": "system", "content": prompt})
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.to(model.device),
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            temperature=0.3,
            repetition_penalty=1.15,
            top_p=0.95,
            do_sample=True,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )
    if return_output:
        return outputs
```

## 英語で試す

はじめは試しに英語をテストします。

```python
ask("Give me a short answer. What are the first 10 numbers in the fibonacci sequence?")
```

    The first ten numbers of the Fibonacci sequence are: 0, 1, 1, 2, 3, 5, 8, 13, 21, and 34.</s>

```python
ask("How did the first President of the United States become a President?")
```

    The first President of the United States, George Washington, became president through an electoral process. In 1789, he was elected by the Electoral College to serve as the 1st President under the newly ratified U.S. Constitution. The Electoral College is made up of 332 members representing each state in proportion to its population and votes for presidential candidates based on their respective states' popular vote.</s>

## 日本語で試す

```python
ask("りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？")
```

    リンゴを5個取り除いた後、残っているリンゴの数は3個です。</s>

```python
ask("バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？")
```

    与えられた情報から、ボールの値段をx、バットの値段をyとする。すると、次のことがわかっています：
    1. バットの値段は、ボールより1,000円高くなっています。つまり、y = x + 1000 となります。
    2. また、バットとボールの合計費用が1100円であることもわかっています。したがって、y + x = 11000 と書くことができます。

    この連立方程式を解くには、まず1つの方程式からもう一方の変数（ここでは y）を表すようにします。最初の式から、y を x の式で表してみましょう。

    y = x + 1000

    次に、これを二番目の式に代入します:
    (x + 1000) + x = 11000

    同類項をまとめます: 2x + 1000 = 11000

    両辺から 1000 を引きます。
    2x = 9000

    ここで、2で割ります:
    x ≈ 4500

    これでボールの値段がわかったので、バットの値段を求めることができます:
    y = x + 1000
    y = 4500 + 1000
    y = 5500

    最終回答：ボールは 5,

```python
ask("A bat and a ball costs $11 total. The bat is $10 more than the ball. How much is the ball")
```

    The problem can be solved using algebra: let x represent the cost of the ball, then the cost of the bat is 10x + 10 (since it's 10 dollars more expensive). We know that their combined cost is $11, so we have this equation:
    x + (1x + 10) = 11

    Now, we need to solve for x. First, combine like terms on the left side of the equation:
    2x + 10 = 11

    Next, subtract 10 from both sides of the equation:
    2x = 1

    Finally, divide by 2 to find the value of x:
    x = 5/2

    So, the cost of the ball is 5/2 dollars.</s>

```python
ask("引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。")
```

    def fibonacci_k(k):
       # k番目のフィボナッチ数は0番目と1番目のフィボナッチの和です
       return (fibonacci_k-1) + fibonacci_k-2</s>

```python
ask("""下記の英語を日本語に翻訳してください。
English: There were 3 apples and 2 oranges. How many fruits were there in total?""")
```

    日本語: リンゴが3個、オレンジが2個ありました。合計で、果物はいくつありましたか？</s>

```python
ask("""
あなたは友達ボットです。できるだけユーザーが親近感を感じやすいよう接してください。

ユーザー: 今日バイト、クビにされたわー。あー人生の意味って何なんだろうねぇー。
アシスタント:
""")
```

    こんにちは！今日はあなたの一日を明るくするお手伝いができれば幸いです！何かお手伝いできることはありますか？</s>

```python
ask("""
### Question
There was a cookie on the table.
Tom entered the room.
The cookie disappeared.
What was likely to have happened?
""")
```

    1. Tom ate it.
    2. The wind blew and carried it away.
    3. A bird took it in its beak.
    4. An invisible hand moved it somewhere else.
    5. It magically appeared elsewhere.</s>

```python
ask("""
### 質問
テーブルにクッキーがおいてありました。
太郎が部屋に入りました。
クッキーが消えました。
何が起きた可能性が高いですか？
""")
```

    1. 太郎はテーブルからクッキーを盗んだ: これは最も可能性が低いです。なぜなら、テーブルにはまだクッキーが残っているからです。
    2. テーブルの上に何かを置く: これは可能性が低いかもしれません。なぜなら、その行為は、テーブルの上に置かれた他のものによって引き起こされる可能性があるからです。
    3. テーブルの下に何かを置いた: これも可能性が低いかもしれませんが、この場合も、テーブルの上の他のものが原因である可能性があります。
    4. テーブルの上に何かを置いておいた: これは考えられるシナリオですが、もし誰かがそれを取り上げた場合、それはまた別の理由で消えるかもしれません。
    5. テーブルの上で何かをした: これはもっともらしいシナリオです。もし、誰かがテーブルの上で何かをしたら、それが原因でクッキーが消えるかもしれません。
    6. テーブルの下にある電気のスイッチを入れて、それを消した: これは可能性が最も高いです。なぜなら、電気のスイッチを入れることは、物理的にクッキーを取り除くことができるからです。</s>

```python
ask("たこ焼きのレシピを教えてください。")
```

    材料:
    - 小麦粉:
    - ベーキングパウダー:
    - 砂糖:
    - 卵:
    - 牛乳:
    - 水:
    - 塩:
    - 片栗粉:
    - 醤油:
    - みりん:
    - おろしニンニク:
    - お好み焼きソース:
    - おろし生姜:
    - かつおぶし:
    - のり:
    - お好みでわさび:
    - お好みでラー油:</s>

```python
ask("大規模言語モデルについて説明してください。")
```

    大規模言語モデルは、自然言語処理の分野における広範な研究と開発のためのフレームワークです。これらのモデルは通常、多次元データセットから生成され、さまざまなタスクに対して学習された高度な統計的および機械学習アルゴリズムを使用します。

    大規模な言語モデルの主な利点は次のとおりです。

    1. 精度: 大規模な言語モデルは、膨大な量のテキストデータを分析して生成されるため、非常に高い精度で予測することができます。これは、人間の専門家が作成したより低い精度と比較して、より信頼性の高い結果を提供できます。

    2. スケーラビリティ: 大規模な言語モデルは、複数のタスクを同時に実行するように設計されており、同時に大量のテキストを処理できるため、多くのユーザーや企業にとって理想的なソリューションとなります。

    3. 柔軟性: 大規模な言語モデルは、異なる言語、方言、またはジャンルに柔軟に対応できるように設計されています。これにより、特定の地域やコミュニティに限定されるものではなく、グローバルなアプリケーションを作成するための基盤を提供します。

    4. API: APIにより、開発者や組織は独自のアプリケーションやツールを開発し、大規模な言語モデルを使用してそれらを拡張または統合できます。

    5. コンプライアンス: 大規模な言語モデルは、プライバシー保護規制に準拠することが不可欠であり、ユーザーの個人情報が収集、保存、および使用されないようにする必要があります。

    6. セキュリティ: 大規模な言語モデルは、攻撃者が不正に使用したり、悪意のある目的で使用したりするのを防ぐために、厳格なセキュリティ対策が必要です。

    7. パフォーマンス: 大規模な言語

```python
ask("""間違いがないよう、正確に質問に答えてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。夏菜子には何人の子供がいますか？
""")
```

    この質問は、2人の兄弟（太郎と二郎）と、その兄弟の母親である夏菜子の間に何人の子供がいるか尋ねています。

    答えは12人です。

    説明：
    1. 次郎は2歳です。
    2. 次郎は弟です。
    3. 次郎は5歳です。
    4. 次郎は4歳です。
    5. 次郎は3歳です。
    6. 次郎は2歳です。
    7. 次郎は1歳です。
    8. 次郎は0歳です。
    9. 次郎は生後6ヶ月です。
    10. 次郎は生後5年です。
    11. 次郎は生後4年です。
    12. 次郎は生後3年です。
    13. 次郎は生後2年です。
    14. 次郎は生後1年です。
    15. 次郎は生後0ヶ月です。
    16. 次郎は生後3ヶ月です。
    17. 次郎は生後2ヶ月です。
    18. 次郎は生後1ヶ月です。
    19. 次郎は生後0日です。
    20. 次郎は生後12日です。
    21. 次郎は生後13日です。
    22. 次郎は生後11日です。
    23. 次郎は生後10日です。
    24. 次郎は生後9日です。
    25. 次郎は生後8日です。
    26. 次郎は生後7日

```python
ask("""下記の文章における登場人物とその関係性をJSON形式でアウトプットしてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。
""")
```

    ```json
    {
     "characters": {
       "Taro": {
         "age": 29,
         "relationships": ["Shinobu", "Sumika"]
       },
       "Shinnosuke": {
         "age": 24,
         "parents": ["Shinobu"],
         "children": [],
         "relationships": ["Taro"]
       },
       "Sumika": {
         "age": 36,
         "relationships": []
       }
     }
    }
    ```</s>

```python
ask("""下記の文章をUTF-8形式のJSON形式でアウトプットしてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。
""")
```

    {
     "family": {
       "members": ["太郎", "二郎"],
       "relationships": {
         "兄妹": true,
         "母子": false,
         "父子": false
       }
     },
     "age": 24,
     "mother": "夏菜子"
    }</s>

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
     "age": 29,
     "family": ["次郎"]
    }</s>

## おわりに

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Colab:
https://colab.research.google.com/drive/1pKQ7XfFsazuSSMnPrWsC5BjH1MGBUQt1?usp=sharing

## 参考

- https://note.com/npaka/n/n24c44bc4bfd6

## 関連

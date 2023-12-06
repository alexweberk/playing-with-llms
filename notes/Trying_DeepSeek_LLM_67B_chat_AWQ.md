# 最近公開された大規模言語モデル DeepSeek-LLM-67B-chat を Colab で試してみた

今回は、DeepSeek-LLM-67B-chat を Colab で試してみたいと思います。
中々大きなモデルサイズなので Colab といっても無料枠では厳しく、今回は A100 の GPU 付きで実行しました。
また、モデルは AWQ 化されたものを使用します。

- 元のモデル: https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat
- AWQ 版: https://huggingface.co/TheBloke/deepseek-llm-67b-chat-AWQ
- ライセンスについて: 商用利用可能な独自ライセンス https://github.com/deepseek-ai/deepseek-LLM/blob/main/LICENSE-MODEL

## セットアップ

初め AWQ 化されたモデルの実行に手こずりました。Colab では CUDA のバージョンが 11.X 系なので、 AWQ も Github から直接 11 系に対応したバージョンをダウンロードすることで動きました。

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
!pip install transformers>=4.35.0 accelerate -Uqq
!pip install https://github.com/casper-hansen/AutoAWQ/releases/download/v0.1.6/autoawq-0.1.6+cu118-cp310-cp310-linux_x86_64.whl -Uqq
```

### モデルのロード

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name_or_path = "TheBloke/deepseek-llm-67b-chat-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)

generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
}

# Using the text streamer to stream output one token at a time
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
```

    AI, or Artificial Intelligence, refers to the ability of a machine or computer program to perform tasks that would normally require human intelligence. This can include things like understanding natural language, recognizing patterns and images, making decisions based on data, and even learning from experience. There are many different types of AI, ranging from simple rule-based systems to complex neural networks that can learn and adapt over time. Some common applications of AI include virtual assistants like Siri and Alexa, recommendation engines used by companies like Netflix and Amazon, and self-driving cars. However, there is ongoing debate about the ethical implications of developing increasingly advanced forms of AI, particularly as it relates to issues such as privacy, bias, and job displacement.

## モデルを試す

簡単に実行できるよう関数を作って実行していきたいと思います。
まずは英語の出力を確認し、その後日本語力を試していきます。

```python
def ask(prompt: str) -> str:
    prompt_template=f'''User: {prompt}

    Assistant:
    '''

    # Convert prompt to tokens
    tokens = tokenizer(
        prompt_template,
        return_tensors='pt'
    ).input_ids.cuda()

    # Generate streamed output, visible one token at a time
    generation_output = model.generate(
        tokens,
        streamer=streamer,
        **generation_params
    )

ask("Give me a short answer. What are the first 10 numbers in the fibonacci sequence?")
```

    1, 1, 2, 3, 5, 8, 13, 21, 34, 55

```python
ask("How did the first President of the United States become a President?")
```

    1. Research about George Washington, the first President of the United States
    2. Identify key events and decisions that led to his presidency
    3. Understand the political climate during the time he was elected
    4. Examine the role of influential individuals in shaping public opinion towards him
    5. Analyze how Washington's personal qualities and achievements contributed to his election
    6. Summarize the factors that made him an ideal candidate for the presidency
    7. Write the answer

    The first President of the United States, George Washington, became president through a combination of his personal qualities, achievements, and historical circumstances surrounding the birth of the nation. Here is how it happened:

    Step 1: Establish credibility and reputation
    George Washington had already built a strong reputation as a military leader during the French and Indian War (1754-1763) and later as the commander-in-chief of the Continental Army during the American Revolutionary War (1775-1783). His dedication, courage, and leadership skills earned him widespread respect among the colonists.

    Step 2: Contribute to the formation of a new government
    After the war ended with the Treaty of Paris in 1783, Washington retired from the army but remained involved in politics. He played an active role in drafting the Articles of Confederation (the original constitution of the US), which established a weak central government over the newly independent states.

    Step 3: Participate in the Constitutional Convention
    In 1787, Washington chaired the Philadelphia Convention, where delegates gathered to revise the Articles of Confederation. Instead, they created a brand-new document called the Constitution, designed to establish a stronger federal government while protecting individual liberties. This event significantly increased Washington's influence on the direction of the young country.

    Step 4: Gain support from influential figures
    Washington maintained close relationships with other founding fathers like Thomas Jefferson, James Madison, John Adams, and Alexander Hamilton. These connections were crucial in building consensus around the Constitution and securing its ratification by the necessary number of states.

    Step 5: Emerge as a unifying figure
    During this period, various factions emerged within the fledgling democracy. However, Washington's reputation and popularity transcended these divisions, making him a widely accepted choice for the presidency. Many saw him as someone who could unite the disparate interests of the different regions and groups within the country.

    Step 6: Accept nomination

### 日本語で試す

```python
ask("りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？")
```

    3つのりんごが残ります。

```python
ask("バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？")
```

    最初に、問題文から得られる情報をまとめてみましょう。
        ・バットとボールを買うと1100円です。
        ・バットはボールよりも1000円高いです。

        これらの情報から、以下の2つの式を導き出すことができます。
        ①　バットの値段 + ボールの値段 = 1100円
        ②　バットの値段 - ボールの値段 = 1000円

        ここで、①式からボールの値段を求めたいので、②式を使って代入することができます。

        その前に、②式を整理して、ボールの値段 = バットの値段 - 1000円と書き換えることができます。

        そして、この値を①式に代入すると、バットの値段 + (バットの値段 - 1000円) = 1100円となります。

        これを整理すると、2 * バットの値段 - 1000円 = 1100円となります。

        さらに整理すると、バットの値段 = (1100円 + 1000円) / 2 = 600円となります。

        最後に、バットの値段は600円であることから、ボールの値段は600円 - 1000円 = -4

```python
ask("A bat and a ball costs $11 total. The bat is $10 more than the ball. How much is the ball")
```

    Let's assume that the cost of the ball is x dollars.
    According to the problem, the bat costs $10 more than the ball, so the cost of the bat would be x + 10 dollars.
    The total cost of the bat and the ball together is $11, so we can write an equation as follows:
    x (ball) + x + 10 (bat) = 11
    Combining like terms, we get:
    2x + 10 = 11
    Subtracting 10 from both sides of the equation, we get:
    2x = 1
    Dividing both sides of the equation by 2, we find that:
    x = 0.5
    Therefore, the cost of the ball is $0.5.

日本語では解けないが、英語だと解けるようです。

```python
ask("引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。")
```

    次のようなPython関数を使用することができます。

        ```python
        def fibonacci(n):
            if n <= 0:
                return "入力は正の整数でなければなりません"

            elif n == 1:
                return [0]

            elif n == 2:
                return [0, 1]

            else:
                fib_sequence = [0, 1]
                while len(fib_sequence) < n:
                    fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])

                return fib_sequence[n-1]

        print(fibonacci(6)) # 5番目のフィボナッチ数列を表示
        ```

        この関数は、与えられた整数`n`に基づいて、`n`個の要素を持つフィボナッチ数列を作成し、その最後の値を返します。最初の2つの要素（0から始まる）は常に同じです。それ以降、新しい要素は直前の2つの要素の和になります。

```python
ask("""下記の英語を日本語に翻訳してください。
English: There were 3 apples and 2 oranges. How many fruits were there in total?""")
```

    英語：There were 3 apples and 2 oranges. How many fruits were there in total?

        日本語：りんごが３個、みかんが２個あります。合計で何個の果物があるでしょうか？

```python
ask("""
あなたは友達ボットです。できるだけユーザーが親近感を感じやすいよう接してください。

ユーザー: 今日バイト、クビにされたわー。あー人生の意味って何なんだろうねぇー。
アシスタント:
""")
```

    申し訳ありませんが、聞こえませんでした。どんな状況か教えて下さい？

```python
ask("""
### Question
There was a cookie on the table.
Tom entered the room.
The cookie disappeared.
What was likely to have happened?
""")
```

    1. Analyze the given information: A cookie was initially present on the table, and after Tom's entrance into the room, it vanished. We are asked to determine what probably occurred based on these facts.

    2. Consider different scenarios that could explain the disappearance of the cookie in relation to Tom entering the room. These possibilities include:
       - Tom taking the cookie.
       - Someone else (not mentioned) took the cookie while Tom wasn't looking or before he entered.
       - The cookie fell off the table accidentally.
       - The cookie was never there in the first place, perhaps an optical illusion or miscommunication.

    3. Evaluate each scenario for likelihood considering available information:
       - If Tom took the cookie, this would be a reasonable assumption since he is one of the two factors mentioned explicitly (the other being the cookie). However, we do not know Tom's character or intentions, so we cannot definitively say if he did take it or not. But, as a person involved directly with both events, his actions might be more closely connected to the outcome than anyone else. Therefore, this possibility should hold some weight.
       - As for someone else taking the cookie, we don't have any information about another individual being present during either event. Thus, without further data, attributing blame (or credit) to an unnamed third party seems speculative at best.
       - Similarly, assessing whether the cookie simply fell off the table may also seem plausible; however, again, no evidence supports or refutes this claim.
       - Finally, if the cookie were never actually there, then its supposed disappearance makes little sense logically speaking—unless we consider possible hallucinations or misunderstandings which would require additional context clues beyond those provided here.

    4. Based on our analysis above, let us weigh each option according to how well they fit within known parameters established by existing details:
       - Tom taking the cookie appears most probable due mainly because it connects him directly with both key events described ("cookie on table" & "Tom enters"). Though we still lack certainty regarding his motives/actions alone, such connections render this hypothesis relatively stronger when compared against alternatives like unknown parties or external factors influencing outcomes indirectly via unrelated circumstances (e.g., gravity causing cookies falling off tables).

    5. Conclusion: Given all available evidence up until now, it is most likely that Tom took the cookie from the table upon entering the room.

```python
ask("""
### 質問
テーブルにクッキーがおいてありました。
太郎が部屋に入りました。
クッキーが消えました。
何が起きた可能性が高いですか？
""")
```

    1. 太郎が食べた可能性：太郎が部屋に入ると、目の前に置いてあるクッキーも見つかったことがあり、彼はそれを食べてしまいました。
        2. 他の誰かが持ち出した可能性：太郎が部屋に入る前に、別の人が通り過ぎていたときに、クッキーが見つかって取られた可能性もあります。
        3. 猫や犬などの動物が盗んだ可能性：貓や犬などの動物が家にいる場合、彼らがクッキーを盗むことがあります。
        4. 風邪患者が咳払い飛ばした可能性：病気中の人が、咳嗽する際に強烈な風邪によって、クッキーが吹っ飛んだ可能性もあります。

```python
ask("たこ焼きのレシピを教えてください。")
```

    タコ焼き（章魚燒）は日本の庶民的な屋台料理です。以下に簡単な作り方を示します。

    材料：
    - 小麦粉 100g
    - 水 250cc
    - 天かす (天婦羅屑) 適量
    - 玉ねぎ (洋蔥) 小さじ1杯分
    - 塩、コショウ、味醂 各小さじ1杯分
    - 柴魚片、青海苔、照り焼きソース、タレ 適量 (トッピング用)

    具材：
    - タコ焼き器

    手順：
    1. 小麦粉と水を混ぜ合わせ、溶けるまで繰り返し混ぜる。
    2. 玉ねぎを細かく刻んで加え、塩、コショウ、味醂を加えて混ぜる。
    3. タコ焼き器に油を塗り、中火で熱し、タコ焼き器に粉漿を注入する。
    4. 粉漿が固まったら、天かすを入れ、転がし、粉漿が流れ込む様子を見ながら、完成形にする。
    5. 熟成されたタコ焼きに、柴魚片、青海苔、照り焼きソース、タレを振りかけ、完成！

    注意点：
    - タコ焼き器は事前に温めておくと、均一に熱くなりやすいです。
    - 火力は弱めに保つことで、外側が焦げずに中身が充分に調理されるよ�

```python
ask("大規模言語モデルについて説明してください。")
```

    大規模言語モデルとは、大量の文書を学習し、自然言語処理（NLP）タスクで使用される深層学習モデルのことです。特徴的なものとしてGPT-3やBERTが挙げられます。大規模言語モデルは、次のような機能を持っています：
        1. 分かりやすさ高い文章生成能力
        2. 様々な領域の知識を活用する事が可能
        3. 人間の感情や意図を理解する事が可能
        4. 自動翻訳や質問応答等の高度なNLPタスクを実現するために利用される

```python
ask("""間違いがないよう、正確に質問に答えてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。夏菜子には何人の子供がいますか？
""")
```

    太郎は二郎より5歳年上であるため、太郎は24 + 5 = 29歳です。二郎と太郎両方が夏菜子さんの子供であることを考慮すると、夏菜子さんには二人の子供がいます。

```python
ask("""下記の文章における登場人物とその関係性をJSON形式でアウトプットしてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。
""")
```

    この文章から、以下のJSONフォーマットで登場人物とその関係性を表現することができます。ただし、登場人物の名前を英語化しています。
    ```json
    {
      "Taizo": {
        "relationships": [
          {"name": "Jiro", "type": "sibling"}
        ],
        "age_difference": {"with": "Jiro", "years": 5}
      },
      "Jiro": {
        "relationships": [
          {"name": "Taizo", "type": "sibling"},
          {"name": "NatsukaChatani", "type": "mother"}
        ],
        "age": 24
      },
      "NatsukaChatani": {
        "relationships": [
          {"name": "Jiro", "type": "child"}
        ]
      }
    }
    ```

```python
ask("""下記の文章をUTF-8形式のJSON形式でアウトプットしてください。

太郎と二郎は兄妹です。太郎は二郎より５つ年上です。夏菜子は二郎の母親です。二郎は２４歳です。
""")
```

    以下は、与えられた文章を表す可能なデータ構造の例です。この回答で使用する言語はPythonですが、他の多くのプログラミング言語にも同様の方法が存在します。また、JSON形式の文字列表現を生成する関数も提供します。

    ```python
    import json

    # 定義されている人物
    taro = {"name": "太郎", "age": None, "sibling": "二郎"}
    jiro = {"name": "二郎", "age": 24, "siblings": ["太郎"]}
    natsukozi = {"name": "夏菜子", "relation_to": "二郎"}

    # Taro is older than Jiro by 5 years
    taro["age"] = jiro["age"] + 5

    data = [taro, jiro, natsukozi]

    def to_json(data):
        return json.dumps(data, ensure_ascii=False)

    print(to_json(data))
    ```

    実行結果：
    ```json
    [{"name": "太郎", "age": 29, "sibling": "二郎"}, {"name": "二郎", "age": 24, "siblings": ["太郎"]}, {"name": "夏菜子", "relation_to": "二郎"}]
    ```

    注意：このJSON形式では、「太郎」は「二郎」の兄弟であり、「夏菜子」は「二郎」の母親であることを示しています。一方で、「二郎」は何人の兄弟姉であるか不明確であるため、配列に値が含まれていません。

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

    以下が、与えられた文章から作成された JSON 形式です。
    ```json
    [
        {
            "name": "太郎",
            "age": 29,
            "family": [
                {"name": "二郎", "age": 24},
                {"name": "夏菜子", "age": null}
            ]
        },
        {
            "name": "二郎",
            "age": 24,
            "family": [
                {"name": "太郎", "age": 29},
                {"name": "夏菜子", "age": null}
            ]
        },
        {
            "name": "夏菜子",
            "age": null,
            "family": [
                {"name": "太郎", "age": 29},
                {"name": "二郎", "age": 24}
            ]
        }
    ]
    ```

## おわりに

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Colab:
https://colab.research.google.com/drive/1GW1k0J7utguhAbpIm55Yq-_iBoiOMad2

## 参考

- 元のモデル: https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat
- AWQ 版: https://huggingface.co/TheBloke/deepseek-llm-67b-chat-AWQ
- ライセンスについて: 商用利用可能な独自ライセンス https://github.com/deepseek-ai/deepseek-LLM/blob/main/LICENSE-MODEL

## 関連

-

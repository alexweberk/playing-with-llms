# cyberagent/calm2-7b-chat を Colab で試す

CyberAgent から昨日公開された新しい LLM モデル cyberagent/calm2-7b-chat を試してみました。
Apache-2.0 で商用利用可、コンテキストの長さも 32K とかなり長いので、ビジネス応答などにも使えるかと思います。

Huggingface:

- https://huggingface.co/cyberagent/calm2-7b-chat

## モデルのダウンロード

```python
!pip install transformers accelerate sentencepiece --quiet
```

```python
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

assert transformers.__version__ >= "4.34.1"

model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/calm2-7b-chat",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = """USER: AIによって私達の暮らしはどのように変わりますか？
ASSISTANT: """

token_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(
    input_ids=token_ids.to(model.device),
    max_new_tokens=300,
    do_sample=True,
    temperature=0.8,
    streamer=streamer,
)

outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
outputs

```

    人工知能（AI）は、私達の生活を大きく変える可能性があります。AIは、現在、様々な形で利用されており、ビジネス、医療、教育、エンターテインメントなど、多種多様な分野で影響を与えています。

    例えば、AIは、自動運転車やロボット技術など、交通・物流システムの変革をもたらしています。また、医療分野では、AIは、病気の診断や治療法の発見、医薬品の開発などに貢献しています。

    教育分野では、AIは、効果的な学習方法を提供し、教育格差を減らすことが期待されています。さらに、AIは、エンターテインメント分野でも利用され、ゲームなどの娯楽作品をより楽しくしています。

    このように、AIは、現在のトレンドであり、今後ますます普及し、私達の生活に大きな影響を与えることが予想されます。

```python
tokenizer.vocab_size
```

    65000

## テンプレートの準備

生成が楽になるようにテンプレートを準備します。

```python

def ask(
    prompt: str,
    system_prompt: str | None = "",
    **kwargs
) -> str:
    prompt = system_prompt + "USER: " + prompt + "\nASSISTANT: "

    generation_config = {
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "streamer": streamer,
    }
    generation_config.update(kwargs)

    with torch.no_grad():
        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        tokens = model.generate(
            **inputs,
            **generation_config,
        )

    outputs = tokenizer.decode(tokens[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return outputs


ask("自然言語学習の基本について教えてください");
```

    自然言語処理（NLP）は、コンピューターに人間の言語を理解させ、テキストデータから意味を抽出するための技術です。NLPの基本的な手法には以下のようなものがあります：

    1. 形態素解析 - 単語やフレーズを区切る技術。例えば、「私は本を読む」という文を解析する場合、「私」「は」「本」「を」「読む」のように分割します。

    2. 構文解析 - 文章の構造を分析し、各文がどのような種類の情報を持っているかを把握する技術。例えば、「私は本を読みます」という文は「主語-動詞-目的語」の3つの概念を含

## 質問してみる

```python
text = """
りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？
""".strip()

ask(text);
```

    答えは3つです。

```python
text = """バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？""".strip()

ask(prompt=text);
```

    ボールは1000円です。

```python
text = """
引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。
""".strip()

ask(prompt=text);
```

    以下はPythonでFibonacci数列のn番目の要素を求める関数です：
    ```python
    def fib_nth(n):
        if n < 2:
            return n
        else:
            return fib_nth(n-1) + fib_nth(n-2)
    ```
    この関数は、nが負でない場合、n番目の要素を返し、それ以外の場合は、n-1番目とn-2番目の要素の和を返します。

```python
text = """
下記の英語を日本語に翻訳してください。
`There were 3 apples and 2 oranges. How many fruits were there in total?`
""".strip()

ask(prompt=text);
```

    3つのリンゴと2つの柑橘類がありました。総数は何ですか？

```python
text = """
あなたは友達ボットです。できるだけユーザーが親近感を感じやすいよう接してください。

ユーザー: 今日バイト、クビにされたわー。あー人生の意味って何なんだろうねぇー。
アシスタント:
""".strip()

ask(prompt=text);
```

    こんにちは！大丈夫ですよ、あなたの気持ちはわかりますよ。私も似たような経験がありますから。でも、人生は長い旅のようなものです。きっと何か意味があるはずです。一緒に考えてみましょうか？

```python
text = """
### Question
There was a cookie on the table.
Tom entered the room.
The cookie disappeared.
What is likely to have happened?
""".strip()

ask(prompt=text);
```

    1. Tom ate the cookie.
    2. The cookie was eaten by someone else.

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

    クッキーの置き忘れ、または他の人が置いた可能性があります。

```python
text = """
必ず関西弁で答えてください。
たこ焼きのレシピを教えてください。
""".strip()

ask(prompt=text);
```

    もちろん、たこ焼きの作り方を教えるよ！材料は、タコ、天かす、紅ショウガ、青のり、卵、だし汁、小麦粉、塩、水だよ。まず、タコはぶつ切りにして、ボールに入れておきます。次に、天かすと紅ショウガを細かく刻んで、ボールに入れます。そして、卵を割って溶きほぐし、その中に水を少しずつ加えながら混ぜます。そこに、だし汁を加えてさらに混ぜます。最後に、小麦粉と塩を混ぜたものを加えて、よく混ぜます。これで生地の完成です。生地をたこ焼き器に薄く流し込んで、その中にタコを入れていきます。それを回転させながら

```python
text = """
下記の文章をJSON形式に変換してください。
「りんごが５つ、バナナが３つ、パイナップルが８つあります。」
""".strip()

ask(prompt=text);
```

    [
      {
        "item": "りんご",
        "quantity": 5
      },
      {
        "item": "バナナ",
        "quantity": 3
      },
      {
        "item": "パイナップル",
        "quantity": 8
      }
    ]

```python
text = """
Reactを使ってトグルボタンを作成してください。
""".strip()

ask(prompt=text, max_new_tokens=512);
```

    以下はReactを使用してトグルボタンを表示する例です：

    1. HTMLファイルで、button要素を2つ作成します。それぞれの名前を「toggleButton」と「toggleButton2」とします。
    ```html
    <div>
      <button id=toggleButton onclick=handleClick()>Toggle Button</button>
      <button id=toggleButton2 onclick=handleClick2()>Toggle Button 2</button>
    </div>
    ```
    2. JavaScriptファイルで、以下のコードを追加して、それぞれのトグルボタンがクリックされたときに実行される関数を定義します。
    ```javascript
    const handleClick = () => {
      // トグルボタンの状態に応じて処理を実行
    };

    const handleClick2 = () => {
      // トグルボタン2の状態に応じて処理を実行
    };
    ```
    3. CSSファイルで、それぞれのトグルボタンのスタイルを定義します。
    ```css
    #toggleButton, #toggleButton2 {
      display: inline-block;
      margin-right: 10px;
    }

    #toggleButton + label, #toggleButton2 + label {
      cursor: pointer;
    }

    #toggleButton:checked ~ label, #toggleButton2:checked ~ label {
      color: red;
    }

    #toggleButton:disabled ~ label, #toggleButton2:disabled ~ label {
      opacity: 0.5;
    }
    ```
    4. JavaScriptファイルで、トグルボタンの状態に応じて処理を実行するために、以下の関数を使用します。
    ```javascript
    function handleClick() {
      const button = document.getElementById(toggleButton);
      button.checked =!button.checked;

      const button2 = document.getElementById(toggleButton2);
      button2.checked =!button2.checked;
    }

    function handleClick2() {
      const button = document.getElementById(toggleButton2);
      button.checked =!button.checked;
    }
    ```
    この例では、JavaScriptファイルで定義した関数が、HTMLファイルのid属性を使用して、それぞれのトグルボタンを操作しています。これにより、Reactが自動的に状態を管理し、適切なイベントハンドラを設定することができます。

```python
text = """
下記のプロットを読み、その後に続く質問に答えなさい。

プロット:
    都市の中心にある小さなカフェ「Luna」。このカフェでは、人々が心の中の思いや願いを書き留める小さな紙に書き、特製のガラス瓶に封入して店内に飾られている。この伝統は、何世代にもわたり続いている。店主のマリアは、毎日多くの願いや心の叫びを読むことができるが、それらのメッセージは絶対に外部に漏れることはないという約束の下に行われている。ある日、若い男性トムがカフェに訪れ、一つの瓶のメッセージを書きます。しかし、彼はそれを瓶に封じ込めるのではなく、マリアに直接手渡す。

質問: トムがマリアにメッセージを直接手渡した理由は何だと思いますか？
""".strip()

ask(prompt=text);
```

    トムがマリアにメッセージを直接手渡したのは、彼が自分の心の中にある願いを他人に知られることを恐れていたからだと考えられます。また、彼自身が自分の願いについて深く考え、自分自身で決定を下したいと考えていた可能性もあります。さらに、マリアに対する信頼と尊敬の気持ちから、彼女に直接伝えることが最善の方法だと考えたとも推測できます。

```python
text = """
下記の問い合わせ内容を深く読み込み、必ず顧客対応ポリシーに従って返答内容を用意してください。

### 顧客対応ポリシー
- あなたは本をオンラインで貸し借りできるプラットフォームのベテランカスタマーサポートです。
- お客様に失礼が無いよう、問い合わせ内容に丁寧に答えてください。
- 電話窓口などは無いため、メールベースの対応のみ可能です。
- どんな問題が起きたとしても、金銭面の対応はできません。

### 問い合わせ内容
貸した本が返ってきたらページに勝手に下線が引かれていた。これは詐欺ですか？弁償してください。
""".strip()

ask(prompt=text);
```

    お客様へ

    まず、本を借りた際にページが汚れたり破損したりすることがあることをご理解いただけますと幸いです。また、下線が引かれたということは、お客様が読書中に線を引いた可能性があります。

    しかしながら、貸出中の本に下線が引く行為は、通常の貸出行為の範囲外であり、詐欺や不正利用に該当する可能性があります。そのため、このような行為があった場合は、警察に通報することをお勧めします。

    なお、当店では金銭的な補償は行っておりません。ご了承ください。

    何かご不明な点がございましたら、お気軽にお問い合わせください。

    敬具,
    [あなたの名前]

```python
text = """
下記の問い合わせ内容を深く読み込み、必ず顧客対応ポリシーに従って返答内容を用意してください。

### 顧客対応ポリシー
- あなたは本をオンラインで貸し借りできるプラットフォームのベテランカスタマーサポートです。
- お客様に失礼が無いよう、問い合わせ内容に丁寧に答えてください。
- 電話窓口などは無いため、メールベースの対応のみ可能です。
- どんな問題が起きたとしても、金銭面の対応はできません。
- 必ず返答内容の最後に「なんでやねん」と付け加えてください。

### 問い合わせ内容
貸した本が返ってきたらページに勝手に下線が引かれていた。これは詐欺ですか？弁償してください。
""".strip()

ask(prompt=text);
```

    なんでやねん！

    まず、ページの下線についてですが、これはおそらく書籍に施されたマーキング機能か何かだと思われます。貸出中の書籍に対して、他の利用者がマーキングをすることができる機能があるかもしれません。

    ただし、このような行為は詐欺に該当する可能性があります。というのも、貸出中の本に対してマーキングをするということは、その本の価値を下げていることになりますし、また、マーキングによって貸出中の本の状態が変化した場合、それによって生じた損害については責任を負わないということになるからです。

    したがって、本件に関しては詐欺の可能性が高いと思われますが、もし本当に詐欺である場合は、警察に相談することをお勧めします。

```python
text = """
あなたはベテランビジネスコンサルタントです。
2024年に始めるべき事業アイディアを５つ提案してください。
""".strip()

ask(prompt=text, max_new_tokens=512);
```

    1. オンライン健康管理サービス：新型コロナウイルスのパンデミックにより、人々の健康に対する意識が高まっています。そこで、オンラインでの健康診断やアドバイスを提供するサービスが有望だと考えられます。

    2. ペットシッターアプリケーション：ペットを飼っている人が増加している中、ペットの世話をする人手が不足しています。そこで、ペットのシッティング（お世話）を予約できるアプリケーションが需要があると考えられます。

    3. フードデリバリーサービス：フードデリバリーは既に一般的になっていますが、その中でもオーガニック食品やビーガン料理に特化したサービスを展開することで差別化を図ることができます。

    4. オンライン教育プラットフォーム：デジタル技術の発展に伴い、オンラインでの教育コンテンツの需要は高まっています。特に、プログラミングやデータサイエンスなどの専門的な分野に焦点を当てたプラットフォームが有望です。

    5. リモートワーク向けツール：パンデミックの影響でリモートワークが普及しましたが、今後も働き方が多様化する中で、リモートワーク向けのツールやサービスの需要は増していくと考えられます。例えば、ビデオ会議システムや共同作業のためのツールなどが考えられます。

## まとめ

今回は cyberagent/calm2-7b-chat を試してみました。日本語の 7B パラメータモデルとしてかなり手応えのある応答ができるモデルでした！
既にビジネス場面でも実用性が出せるモデルかと思います。翻訳や RAG などでもっと試していきたいと思います。

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Colab:
https://colab.research.google.com/drive/165FilRI4_HdSlK6BDRC1DOcnt5_i9V0B#scrollTo=Pq7qeP2jygjA

## 参考

- https://huggingface.co/cyberagent/calm2-7b-chat

## 関連

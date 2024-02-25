# 最新の Google Gemma モデルを MLX を使ってローカルでファインチューニング

今回は、最新の [Google Gemma モデル](https://blog.google/technology/developers/gemma-open-models/)を Apple Silicon に最適化されたライブラリ `MLX` を使ってローカルで実行したり、ファインチューニングしてみましたのでその手順を紹介します。
MLX 関連の情報はドキュメンテーションが分かりづらいものも多かったので色々試した経緯も共有しながら少しでも何かの参考になれば幸いです。

実際に使った Jupyter Notebook を Gist にアップロードしていますので、そちらも参考にしてください。

- [Google Gemma モデルを MLX を使ってローカルでファインチューニング](https://gist.github.com/alexweberk/1434c95c05463866491677aac6ce19ba)

## 事前準備

必要なライブラリをインストールします。
また Apple Silicon 搭載の Mac が必要です。今回は M3 Max 128GB 搭載の MacBook Pro で実行しました。

```python
!pip install -U mlx mlx_lm transformers
```

## mlx_lm を使ったモデルの実行

公開された Gemma には 4 つほどバージョンがありますが、今回は instruction チューニング済みの `gemma-7b-it` を使ってみました。

mlx バックエンドを活用した `mlx_lm` ライブラリを使います。

```python
from mlx_lm import generate, load

model, tokenizer = load("google/gemma-7b-it")
```

一応英語を中心としたモデルなので、まずは英語が問題なく生成できるか試してみます。

MLX での生成の場合、プロンプトテンプレートをどうすればよいのか検索してもわかりませんでした。
ただ、 `mlx-examples` のリポのコードを読む限りは `transformers` の tokenizer に `apply_chat_template` メソッドがある場合はそれを使ってくれているようでした。
そのため、生成の際には質問内容だけを含んだプロンプトをインプットとします。

https://github.com/ml-explore/mlx-examples/blob/47dd6bd17f3cc7ef95672ea16e443e58ce5eb1bf/llms/mlx_lm/generate.py#L98

```python
# ライブラリ内でのテンプレート適用後のプロンプト
messages = [{"role": "user", "content": "Why is the sky blue?"}]
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

    '<bos><start_of_turn>user\nWhy is the sky blue?<end_of_turn>\n<start_of_turn>model\n'

というわけで、実行。

```python
# プロンプトテンプレート無しでの生成
prompt = """
Why is the sky blue?
""".strip()
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,  # Set to True to see the prompt and response
    temp=0.0,
    max_tokens=256,
)
```

    ==========
    Prompt: Why is the sky blue?


    The sky is blue due to a phenomenon called **Rayleigh Scattering**.

    Here's a breakdown of what happens:

    1. **Sunlight:** Sunrays are made up of all the colors of the rainbow, with each color having a different wavelength.
    2. **Scattering:** When sunlight enters Earth's atmosphere, it interacts with the tiny particles of air (dust, water vapor, etc.). These particles scatter the sunlight in all directions.
    3. **Blue Scatter:** The particles scatter the shorter wavelengths of blue and violet light more effectively than the longer wavelengths of red and orange light.
    4. **Scattered Light:** The scattered light, which is predominantly blue, is scattered in all directions.
    5. **Our View:** We see the scattered light from all directions, including the direction opposite the sun. This is why the sky appears blue.

    **Additional factors:**

    * **Time of Day:** The intensity of the blue color is strongest at midday and decreases as the sun gets closer to the horizon.
    * **Clouds:** Clouds can reduce the amount of scattered light, making the sky appear white or gray.
    * **Dust:** Dust particles can also scatter different colors of light, which can affect the appearance of the sky.


    ==========
    Prompt: 16.696 tokens-per-sec
    Generation: 18.644 tokens-per-sec

### 日本語の生成

英語がうまく生成できていそうだったので、次に日本語を生成してみます。

```python
# プロンプトテンプレートなしでの生成
prompt = """
空が青いのはなぜですか？
""".strip()
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,  # Set to True to see the prompt and response
    temp=0.0,
    max_tokens=256,
)
```

    ==========
    Prompt: 空が青いのはなぜですか？


    実際、空気は実際実際赤い色です。ただし、人間の目は赤い色を認識するには、特定の波長の光が必要です。空気の分子は、その特定の波長の光を吸収し、人間の目に届く残り色を青に見えます。
    ==========
    Prompt: 19.429 tokens-per-sec
    Generation: 17.932 tokens-per-sec

```python
# よくあるテンプレートで指示文は英語の場合
prompt = """
## Instructions
You are a helpful AI assistant. Answer questions from the user to the best of your knowledge.
If you don't know the answer, be truthful in your answer.
Always answer in perfectly natural Japanese.

## Questions
空が青いのはなぜですか？
""".strip()
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,  # Set to True to see the prompt and response
    temp=0.0,
    max_tokens=256,
)
```

    ==========
    Prompt: ## Instructions
    You are a helpful AI assistant. Answer questions from the user to the best of your knowledge.
    If you don't know the answer, be truthful in your answer.
    Always answer in perfectly natural Japanese.

    ## Questions
    空が青いのはなぜですか？


    ## Answer
    空が青い理由は、いくつかの原因があります。

    * **空気中の水蒸気:** 空中の水蒸気は、太陽によって加熱され、昇華し、雲が発生します。雲は空気の温度を下げ、空気中の粒子を小さくします。
    * **太陽の光:** 太陽の光は、空気中の分子を励起し、空気の色を変化させます。
    * **視覚の限界:** 人間は、実際よりも多くの色を認識することは不可能です。そのため、実際よりも明るい色や鮮度のある色を感じます。
    ==========
    Prompt: 140.153 tokens-per-sec
    Generation: 18.544 tokens-per-sec

```python
# よくあるテンプレートで指示文も日本語の場合
prompt = """
## 指示文
あなたは役に立つAIアシスタントです。ユーザーからの質問にできる限りの知識で答えます。
答えがわからない場合は、正直な答えをしてください。
解答は必ず自然な日本語で行ってください。

## 質問
空が青いのはなぜですか？
""".strip()
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,  # Set to True to see the prompt and response
    temp=0.0,
    max_tokens=256,
)
```

    ==========
    Prompt: ## 指示文
    あなたは役に立つAIアシスタントです。ユーザーからの質問にできる限りの知識で答えます。
    答えがわからない場合は、正直な答えをしてください。
    解答は必ず自然な日本語で行ってください。

    ## 質問
    空が青いのはなぜですか？


    ## 解答
    空気中の成分や雲々の影響によって、実際は実際は空は実際は青ではなく、実際は黒です。ただし、人間の視覚は、空気中の成分や雲々の影響によって、実際よりも明るい色を認識します。そのため、私たちが見た空は実際よりも明るく、そして青いように感じられます。
    ==========
    Prompt: 144.546 tokens-per-sec
    Generation: 18.491 tokens-per-sec

指示文を英語にしたことで実際英語の生成に近い構造の回答になったのが実際実際面白いですね。

次は MLX を使ってファインチューニングをしてみます。

## MLX を使って Gemma モデルを LoRA ファインチューニング

今回 [npaka さんの記事](https://note.com/npaka/n/nc55e44e407ff?sub_rt=share_h)を参考に「ござるデータセット」を使ってファインチューニングを行いました。
https://huggingface.co/datasets/bbz662bbz/databricks-dolly-15k-ja-gozarinnemon/tree/main

[MLX Examples](https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md#Custom-Data)を参考に、教師データのフォーマットを変えます。

※超絶ややこしいですが、はじめ `mlx-examples/lora` を使おうとして `python lora.py` 等としてましたが、こちらのコードはまだ `gemma` モデルに対応していませんでした。正しくは `mlx-examples/llms/mlx_lm` の方の lora.py 等を参照し、 `python -m mlx_lm.lora ...` 等とする必要があります。

### ファインチューニング用のデータの成形

```python
import pandas as pd

data = pd.read_json("data/databricks-dolly-15k-ja-gozarinnemon.json")
data = data[data["category"] == "open_qa"]
data
```

```python
def generate_prompt(example: pd.Series) -> str:
    "教師データをGemmaプロンプトの形にあわせるための関数"
    return """<bos><start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(
        example["instruction"], example["output"]
    )


data["text"] = data.apply(generate_prompt, axis=1)
print(data["text"].iloc[0])
```

    <bos><start_of_turn>user
    手品がうまくなるにはどうしたらいいのでしょうか？<end_of_turn>
    <start_of_turn>model
    我、りんえもんは思う。 マジックは、何時間も何時間も練習することで上達しますでござる。  マジックは、手の筋肉の記憶によって構築される非常に器用さを必要とするため、カードのデッキを入手し、その持ち方とフラリッシュのやり方を学ぶことは、始めるのに素晴らしい方法でござる。  マジックは、手の筋肉の記憶によって構築される非常に器用さを必要とするため、上手になるには何時間も何時間も繰り返し練習する必要がありますでござる。  マジックが上手になるには、マジックへの興味も重要な要素でござる。  もし興味がなければ、マジックがうまくなることはないでしょう。  また、クラブに参加し、それに興味を持っている他の人々と一緒にマジックのトリックを練習することができますでござる。  参加するのに最適な組織の1つは、アメリカのマジシャン協会（SAM）でござる。知らんけど。<end_of_turn><eos>

トレーニングセットとバリデーションセットに分けてファイルに保存します。

- トレーニングセット: `data/train.jsonl`
- バリデーションセット: `data/valid.jsonl`

`mlx_lm.lora` では train.jsonl と valid.jsonl が必須のようでした。

```python
split_ix = int(len(data) * 0.9)
# shuffle data
data = data.sample(frac=1, random_state=42)
train, valid = data[:split_ix], data[split_ix:]

# Save train and valid dataset as jsonl files
train[["text"]].to_json("data/train.jsonl", orient="records", lines=True, force_ascii=False)
valid[["text"]].to_json("data/valid.jsonl", orient="records", lines=True, force_ascii=False)

!head -n 5 data/train.jsonl
```

    {"text":"<bos><start_of_turn>user\n手品がうまくなるにはどうしたらいいのでしょうか？<end_of_turn>\n<start_of_turn>model\n我、りんえもんは思う。 マジックは、何時間も何時間も練習することで上達しますでござる。  マジックは、手の筋肉の記憶によって構築される非常に器用さを必要とするため、カードのデッキを入手し、その持ち方とフラリッシュのやり方を学ぶことは、始めるのに素晴らしい方法でござる。  マジックは、手の筋肉の記憶によって構築される非常に器用さを必要とするため、上手になるには何時間も何時間も繰り返し練習する必要がありますでござる。  マジックが上手になるには、マジックへの興味も重要な要素でござる。  もし興味がなければ、マジックがうまくなることはないでしょう。  また、クラブに参加し、それに興味を持っている他の人々と一緒にマジックのトリックを練習することができますでござる。  参加するのに最適な組織の1つは、アメリカのマジシャン協会（SAM）でござる。知らんけど。<end_of_turn><eos>"}
    {"text":"<bos><start_of_turn>user\nカエサルはどのようにしてローマで権力を握ったのでしょうか？<end_of_turn>\n<start_of_turn>model\n我、りんえもんは思う。 女神ヴィーナスとトロイの王子アイネアスの血を引くジュリイ家と呼ばれるパトリシアン（元老院議員）一族に生まれた。この家系に生まれたことで、彼は権力への道を他の誰よりも先に歩むことになった。\n\nカエサルは賢く、勤勉で魅力的な人物で、後世に役立つ多くの友人や人脈を作りましたでござる。当時ローマの独裁者であったガイウス・マリウスの親戚と結婚し、政治的なコネクションをさらに深めましたでござる。スッラがマリウスを破ったことで挫折を味わったが、カエサルはそれを逆手にとって、ローマに留まらなければならない赴任を逃れることができた。コンスル（ローマの指導者）を目指す者にとって、ローマのために土地を征服した有能な軍師として見られることは重要なことでござった。\n\nカエサルの次の重要な地位はアエディレで、主な職務のひとつはゲームの企画でござった。カエサルは、ローマの人々に我を知ってもらうために、多額の借金をし、豪華な競技を開催した。カエサルは、他の貴族と違って、当時平民が住んでいたスブラで育ったので、民衆の愛によって得られる重要性と権力を知っていた。\n\nカエサルは何度も任命されながら、その軍事的才能を発揮し、最終的には執政官に選ばれた。カエサルは、当時の有力者2人と同盟を結びました：ポンペイ（名将）、クラッスス（ローマ一の富豪）と同盟を結び、「第一次三頭政治」と呼ばれるようになりましたでござる。\n\nカエサルは領事職の後、ガリア総督となり、ローマに多くの領土を獲得させた。しかし、ローマの政敵たちは、カエサルを訴追し、権力を剥奪するための裁判を準備した。カエサルは軍団の一つを集め、ローマに進軍した。政敵はローマから戦術的に撤退し、カエサルは名目上の支配権を握った。カエサルは、ギリシャ、スペイン、エジプトでの戦争に勝利して権力を強化し、最終的にはローマでの支配を確固たるものにし、独裁者に選出された。知らんけど。<end_of_turn><eos>"}
    {"text":"<bos><start_of_turn>user\nResearch Collaboratory for Structural Bioinformatics Protein Data Bank (RCSB PDB)とは何ですか？<end_of_turn>\n<start_of_turn>model\n我、りんえもんは思う。 Research Collaboratory for Structural Bioinformatics Protein Data Bank (RCSB PDB) は、タンパク質や核酸などの高分子の立体構造について豊富な情報を提供するデータベースでござる。このデータベースには、多数の高分子の実験的に決定された原子座標が含まれており、高分子の構造、機能、相互作用を研究するために利用することができますでござる。RCSB PDBは、多くの生物学的プロセスの構造基盤を理解し、特定の高分子を標的とする新薬を設計するための貴重な資源となるため、ゲノミクス研究や創薬の分野で広く利用されていますでござる。\n\nRCSB PDBには、原子座標に加えて、構造決定に使用した実験方法、関連する科学文献の引用、生物学的機能、配列、相同性に関する情報など、各高分子に関する豊富な追加情報が含まれていますでござる。また、構造データの可視化、解析、ダウンロードのための様々なツールやリソースも提供されており、高分子の3次元構造をインタラクティブに探索できるウェブベースのビューアもありますでござる。知らんけど。<end_of_turn><eos>"}
    {"text":"<bos><start_of_turn>user\nロシア文学の名作をいくつか挙げる<end_of_turn>\n<start_of_turn>model\n我、りんえもんは思う。 * 戦争と平和、レオ・トルストイ。\n* 罪と罰、フョードル・ドストエフスキー（Fyodor Dostoevsky）。\n* エフゲニー・オネーギン、アレクサンドル・プーシキン。\n* 死せる魂、ニコライ・ゴーゴリ。\n* 災い転じて福となす、アレクサンドル・グリボエドフ。\n* マスターとマルガリータ、ミハイル・ブルガーコフ。\n* オブロモフ、イワン・ゴンチャロフ知らんけど。<end_of_turn><eos>"}
    {"text":"<bos><start_of_turn>user\nゴールデンステイト・ウォリアーズで一番好きな選手は誰ですか？<end_of_turn>\n<start_of_turn>model\n我、りんえもんは思う。 AIである我は、好きなバスケットボール選手はいませぬ。しかし、もし選ぶとしたら、ゲイリー・ペイトン2世を選びますでござる。彼はハートを持ってプレーし、執拗にディフェンスし、誰にでもダンクを決める。知らんけど。<end_of_turn><eos>"}

### MLX で LoRA ファインチューニングを実行

`mlx_lm` で LoRA する場合、下記のコマンドでいろんなオプションが出せます。

```python
!python -m mlx_lm.lora --help
```

    usage: lora.py [-h] [--model MODEL] [--max-tokens MAX_TOKENS] [--temp TEMP]
                   [--prompt PROMPT] [--train] [--data DATA]
                   [--lora-layers LORA_LAYERS] [--batch-size BATCH_SIZE]
                   [--iters ITERS] [--val-batches VAL_BATCHES]
                   [--learning-rate LEARNING_RATE]
                   [--steps-per-report STEPS_PER_REPORT]
                   [--steps-per-eval STEPS_PER_EVAL]
                   [--resume-adapter-file RESUME_ADAPTER_FILE]
                   [--adapter-file ADAPTER_FILE] [--save-every SAVE_EVERY]
                   [--test] [--test-batches TEST_BATCHES]
                   [--max-seq-length MAX_SEQ_LENGTH] [--seed SEED]

    LoRA or QLoRA finetuning.

    options:
      -h, --help            show this help message and exit
      --model MODEL         The path to the local model directory or Hugging Face
                            repo.
      --max-tokens MAX_TOKENS, -m MAX_TOKENS
                            The maximum number of tokens to generate
      --temp TEMP           The sampling temperature
      --prompt PROMPT, -p PROMPT
                            The prompt for generation
      --train               Do training
      --data DATA           Directory with {train, valid, test}.jsonl files
      --lora-layers LORA_LAYERS
                            Number of layers to fine-tune
      --batch-size BATCH_SIZE
                            Minibatch size.
      --iters ITERS         Iterations to train for.
      --val-batches VAL_BATCHES
                            Number of validation batches, -1 uses the entire
                            validation set.
      --learning-rate LEARNING_RATE
                            Adam learning rate.
      --steps-per-report STEPS_PER_REPORT
                            Number of training steps between loss reporting.
      --steps-per-eval STEPS_PER_EVAL
                            Number of training steps between validations.
      --resume-adapter-file RESUME_ADAPTER_FILE
                            Load path to resume training with the given adapter
                            weights.
      --adapter-file ADAPTER_FILE
                            Save/load path for the trained adapter weights.
      --save-every SAVE_EVERY
                            Save the model every N iterations.
      --test                Evaluate on the test set after training
      --test-batches TEST_BATCHES
                            Number of test set batches, -1 uses the entire test
                            set.
      --max-seq-length MAX_SEQ_LENGTH
                            Maximum sequence length.
      --seed SEED           The PRNG seed

早速トレーニングしてみましょう。テストのため 600 ステップだけトレーニングします。

```python
!python -m mlx_lm.lora \
    --model google/gemma-7b-it \
    --train \
    --iters 600 \
    --data data
```

    Loading pretrained model
    Fetching 11 files: 100%|█████████████████████| 11/11 [00:00<00:00, 32652.05it/s]
    Total parameters 8539.516M
    Trainable parameters 1.835M
    Loading datasets
    Training
    Starting training..., iters: 600
    Iter 1: Val loss 11.819, Val took 99.751s
    Iter 10: Train loss 11.533, Learning Rate 1.000e-05, It/sec 0.212, Tokens/sec 105.547, Trained Tokens 4969
    Iter 20: Train loss 8.003, Learning Rate 1.000e-05, It/sec 0.050, Tokens/sec 33.970, Trained Tokens 11712
    Iter 30: Train loss 7.366, Learning Rate 1.000e-05, It/sec 0.272, Tokens/sec 128.895, Trained Tokens 16459
    Iter 40: Train loss 5.418, Learning Rate 1.000e-05, It/sec 0.278, Tokens/sec 121.127, Trained Tokens 20809
    Iter 50: Train loss 3.997, Learning Rate 1.000e-05, It/sec 0.305, Tokens/sec 122.895, Trained Tokens 24836
    Iter 60: Train loss 3.450, Learning Rate 1.000e-05, It/sec 0.058, Tokens/sec 40.930, Trained Tokens 31839
    Iter 70: Train loss 3.217, Learning Rate 1.000e-05, It/sec 0.379, Tokens/sec 136.167, Trained Tokens 35433
    Iter 80: Train loss 3.088, Learning Rate 1.000e-05, It/sec 0.318, Tokens/sec 128.851, Trained Tokens 39491
    Iter 90: Train loss 3.062, Learning Rate 1.000e-05, It/sec 0.308, Tokens/sec 130.390, Trained Tokens 43719
    Iter 100: Train loss 2.816, Learning Rate 1.000e-05, It/sec 0.273, Tokens/sec 140.060, Trained Tokens 48851
    Iter 100: Saved adapter weights to checkpoints/100_adapters.npz.
    [WARNING] Some sequences are longer than 2048 tokens. The longest sentence 3640 will be truncated to 2048. Consider pre-splitting your data to save memory.
    Iter 110: Train loss 2.900, Learning Rate 1.000e-05, It/sec 0.049, Tokens/sec 29.810, Trained Tokens 54916
    Iter 120: Train loss 2.812, Learning Rate 1.000e-05, It/sec 0.268, Tokens/sec 117.730, Trained Tokens 59317
    Iter 130: Train loss 2.858, Learning Rate 1.000e-05, It/sec 0.258, Tokens/sec 120.012, Trained Tokens 63962
    Iter 140: Train loss 2.726, Learning Rate 1.000e-05, It/sec 0.285, Tokens/sec 122.977, Trained Tokens 68279
    Iter 150: Train loss 2.836, Learning Rate 1.000e-05, It/sec 0.318, Tokens/sec 139.799, Trained Tokens 72677
    Iter 160: Train loss 2.903, Learning Rate 1.000e-05, It/sec 0.394, Tokens/sec 139.106, Trained Tokens 76209
    Iter 170: Train loss 2.735, Learning Rate 1.000e-05, It/sec 0.308, Tokens/sec 138.257, Trained Tokens 80694
    Iter 180: Train loss 2.844, Learning Rate 1.000e-05, It/sec 0.224, Tokens/sec 107.364, Trained Tokens 85493
    Iter 190: Train loss 2.613, Learning Rate 1.000e-05, It/sec 0.315, Tokens/sec 118.791, Trained Tokens 89263
    Iter 200: Train loss 2.675, Learning Rate 1.000e-05, It/sec 0.308, Tokens/sec 126.078, Trained Tokens 93363
    Iter 200: Val loss 2.770, Val took 49.972s
    Iter 200: Saved adapter weights to checkpoints/200_adapters.npz.
    Iter 210: Train loss 2.598, Learning Rate 1.000e-05, It/sec 0.350, Tokens/sec 120.461, Trained Tokens 96808
    Iter 220: Train loss 2.691, Learning Rate 1.000e-05, It/sec 0.170, Tokens/sec 94.299, Trained Tokens 102342
    Iter 230: Train loss 2.485, Learning Rate 1.000e-05, It/sec 0.278, Tokens/sec 110.115, Trained Tokens 106302
    Iter 240: Train loss 2.565, Learning Rate 1.000e-05, It/sec 0.305, Tokens/sec 122.773, Trained Tokens 110332
    Iter 250: Train loss 2.665, Learning Rate 1.000e-05, It/sec 0.308, Tokens/sec 131.526, Trained Tokens 114605
    Iter 260: Train loss 2.566, Learning Rate 1.000e-05, It/sec 0.270, Tokens/sec 106.033, Trained Tokens 118531
    Iter 270: Train loss 2.703, Learning Rate 1.000e-05, It/sec 0.271, Tokens/sec 109.787, Trained Tokens 122584
    Iter 280: Train loss 2.818, Learning Rate 1.000e-05, It/sec 0.175, Tokens/sec 101.312, Trained Tokens 128358
    Iter 290: Train loss 2.703, Learning Rate 1.000e-05, It/sec 0.314, Tokens/sec 121.601, Trained Tokens 132226
    Iter 300: Train loss 2.659, Learning Rate 1.000e-05, It/sec 0.186, Tokens/sec 98.913, Trained Tokens 137555
    Iter 300: Saved adapter weights to checkpoints/300_adapters.npz.
    Iter 310: Train loss 2.579, Learning Rate 1.000e-05, It/sec 0.321, Tokens/sec 120.959, Trained Tokens 141326
    Iter 320: Train loss 2.519, Learning Rate 1.000e-05, It/sec 0.323, Tokens/sec 117.220, Trained Tokens 144956
    Iter 330: Train loss 2.692, Learning Rate 1.000e-05, It/sec 0.216, Tokens/sec 102.143, Trained Tokens 149685
    Iter 340: Train loss 2.577, Learning Rate 1.000e-05, It/sec 0.331, Tokens/sec 109.429, Trained Tokens 152987
    Iter 350: Train loss 2.519, Learning Rate 1.000e-05, It/sec 0.222, Tokens/sec 102.454, Trained Tokens 157592
    Iter 360: Train loss 2.675, Learning Rate 1.000e-05, It/sec 0.286, Tokens/sec 115.840, Trained Tokens 161638
    Iter 370: Train loss 2.527, Learning Rate 1.000e-05, It/sec 0.278, Tokens/sec 107.560, Trained Tokens 165501
    Iter 380: Train loss 2.544, Learning Rate 1.000e-05, It/sec 0.298, Tokens/sec 108.646, Trained Tokens 169142
    Iter 390: Train loss 2.599, Learning Rate 1.000e-05, It/sec 0.269, Tokens/sec 116.284, Trained Tokens 173457
    Iter 400: Train loss 2.566, Learning Rate 1.000e-05, It/sec 0.268, Tokens/sec 117.563, Trained Tokens 177838
    Iter 400: Val loss 2.636, Val took 56.804s
    Iter 400: Saved adapter weights to checkpoints/400_adapters.npz.
    Iter 410: Train loss 2.338, Learning Rate 1.000e-05, It/sec 0.238, Tokens/sec 97.225, Trained Tokens 181923
    Iter 420: Train loss 2.602, Learning Rate 1.000e-05, It/sec 0.242, Tokens/sec 111.528, Trained Tokens 186539
    Iter 430: Train loss 2.846, Learning Rate 1.000e-05, It/sec 0.227, Tokens/sec 104.883, Trained Tokens 191152
    Iter 440: Train loss 2.562, Learning Rate 1.000e-05, It/sec 0.275, Tokens/sec 115.619, Trained Tokens 195359
    Iter 450: Train loss 2.571, Learning Rate 1.000e-05, It/sec 0.245, Tokens/sec 112.251, Trained Tokens 199934
    Iter 460: Train loss 2.730, Learning Rate 1.000e-05, It/sec 0.212, Tokens/sec 96.232, Trained Tokens 204475
    Iter 470: Train loss 2.491, Learning Rate 1.000e-05, It/sec 0.284, Tokens/sec 103.841, Trained Tokens 208133
    Iter 480: Train loss 2.579, Learning Rate 1.000e-05, It/sec 0.179, Tokens/sec 99.164, Trained Tokens 213682
    Iter 490: Train loss 2.578, Learning Rate 1.000e-05, It/sec 0.259, Tokens/sec 107.788, Trained Tokens 217850
    Iter 500: Train loss 2.411, Learning Rate 1.000e-05, It/sec 0.292, Tokens/sec 111.877, Trained Tokens 221676
    Iter 500: Saved adapter weights to checkpoints/500_adapters.npz.
    Iter 510: Train loss 2.354, Learning Rate 1.000e-05, It/sec 0.307, Tokens/sec 127.103, Trained Tokens 225819
    Iter 520: Train loss 2.547, Learning Rate 1.000e-05, It/sec 0.263, Tokens/sec 110.252, Trained Tokens 230019
    Iter 530: Train loss 2.600, Learning Rate 1.000e-05, It/sec 0.351, Tokens/sec 119.239, Trained Tokens 233417
    Iter 540: Train loss 2.540, Learning Rate 1.000e-05, It/sec 0.313, Tokens/sec 126.378, Trained Tokens 237456
    Iter 550: Train loss 2.464, Learning Rate 1.000e-05, It/sec 0.202, Tokens/sec 100.631, Trained Tokens 242442
    Iter 560: Train loss 2.635, Learning Rate 1.000e-05, It/sec 0.277, Tokens/sec 121.453, Trained Tokens 246827
    Iter 570: Train loss 2.459, Learning Rate 1.000e-05, It/sec 0.180, Tokens/sec 92.789, Trained Tokens 251991
    Iter 580: Train loss 2.571, Learning Rate 1.000e-05, It/sec 0.210, Tokens/sec 101.168, Trained Tokens 256811
    Iter 590: Train loss 2.459, Learning Rate 1.000e-05, It/sec 0.208, Tokens/sec 105.794, Trained Tokens 261886
    Iter 600: Train loss 2.541, Learning Rate 1.000e-05, It/sec 0.278, Tokens/sec 116.746, Trained Tokens 266085
    Iter 600: Val loss 2.578, Val took 54.380s
    Iter 600: Saved adapter weights to checkpoints/600_adapters.npz.
    Saved final adapter weights to adapters.npz.

M3 Max 128GB で約 50 分かかりました。[npaka さんの記事](https://note.com/npaka/n/nc55e44e407ff?sub_rt=share_h)だと 20 分ほどで完了するそうなので、やはり NVIDIA A100 などの GPU と比べてしまうと時間がかかってしまいますね…。

### 推論の実行

コマンドラインの場合は、下記の様なスクリプトで LoRA ウェイトも使った推論ができます。

```python
!python -m mlx_lm.lora --model "google/gemma-7b-it" \
               --adapter-file checkpoints/600_adapters.npz \
               --max-tokens 256 \
               --prompt "空が青いのはなぜですか？" \
            #    --prompt "<bos><start_of_turn>user\n空が青いのはなぜですか？<end_of_turn>\n<start_of_turn>model\n" \
               --seed 69
```

    Loading pretrained model
    Fetching 11 files: 100%|█████████████████████| 11/11 [00:00<00:00, 91542.35it/s]
    Total parameters 8539.516M
    Trainable parameters 1.835M
    Loading datasets
    Generating
    ==========
    Prompt: 空が青いのはなぜですか？
    <end_of_turn>model
    我、りんえもんは思う。 空が青い理由は、分光放射のだろう、放射光と反射光が反射光を遮断し、その反射光を他の色を遮断することで、空が青くなるというござる。  この反射光は、空気の粒子に衝突し、色を反射しますでござる。  空気の粒子に衝突し、その反射光が他の色を遮断することで、空が青くなる。知らんけど。<end_of_turn>
    ==========
    Prompt: 12.347 tokens-per-sec
    Generation: 6.067 tokens-per-sec

## LoRA ウェイトのマージ

最後に、トレーニングした LoRA のウェイトをモデル自体にマージしてみます。

下記のコマンドで用意されているオプションが表示されます。

```python
!python -m mlx_lm.fuse --help
```

    Loading pretrained model
    usage: fuse.py [-h] [--model MODEL] [--save-path SAVE_PATH]
                   [--adapter-file ADAPTER_FILE] [--hf-path HF_PATH]
                   [--upload-repo UPLOAD_REPO] [--de-quantize]

    LoRA or QLoRA finetuning.

    options:
      -h, --help            show this help message and exit
      --model MODEL         The path to the local model directory or Hugging Face
                            repo.
      --save-path SAVE_PATH
                            The path to save the fused model.
      --adapter-file ADAPTER_FILE
                            Path to the trained adapter weights (npz or
                            safetensors).
      --hf-path HF_PATH     Path to the original Hugging Face model. Required for
                            upload if --model is a local directory.
      --upload-repo UPLOAD_REPO
                            The Hugging Face repo to upload the model to.
      --de-quantize         Generate a de-quantized model.

早速マージしてみましょう。

```python
!python -m mlx_lm.fuse \
    --model google/gemma-7b-it \
    --adapter-file checkpoints/600_adapters.npz
```

    Loading pretrained model
    Fetching 11 files: 100%|█████████████████████| 11/11 [00:00<00:00, 91542.35it/s]

無事マージが完了し、 `lora_fused_model` というディレクトリが作成されモデルの各種ファイルが保存されました。

## マージされたモデルのロード

念の為ロードもしてみます。

```python
from mlx_lm import generate, load

fused_model, fused_tokenizer = load("./lora_fused_model/")
```

```python
# 当たり前ですが、tokenizerは変わっていないようです。
fused_tokenizer.apply_chat_template
```

    <bound method PreTrainedTokenizerBase.apply_chat_template of GemmaTokenizerFast(name_or_path='lora_fused_model', vocab_size=256000, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<bos>', 'eos_token': '<eos>', 'unk_token': '<unk>', 'pad_token': '<pad>', 'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
    	0: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	1: AddedToken("<eos>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	2: AddedToken("<bos>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	3: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	106: AddedToken("<start_of_turn>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	107: AddedToken("<end_of_turn>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    }>

```python
prompt = "空が青いのはなぜですか？"
response = generate(
    fused_model,
    fused_tokenizer,
    prompt=prompt,
    verbose=True,  # Set to True to see the prompt and response
    temp=0.0,
    max_tokens=256,
)
```

    ==========
    Prompt: 空が青いのはなぜですか？
    <end_of_turn>model
    我、りんえもんは思う。 空は青色で、その理由は、空の圧力に由来する。空の圧力は、地球の圧力よりも低いので、空の圧力によって圧縮された空気は、地球の圧力によって圧縮された空気よりも圧縮され、その結果として、空気は地球の圧力によって圧縮された空気よりも疎化され、その結果として、空気は地球の圧力によって圧縮された空気よりも疎化され、その結果として、空気は地球の圧力によって圧縮された空気よりも疎化され、その結果として、空気は地球の圧力によって圧縮された空気よりも疎化され、その結果として、空気は地球の圧力によって圧縮された空気よりも疎化され、その結果として、空気は地球の圧力によって圧縮された空気よりも疎化され、その結果として、空気は地球の圧力によって圧縮された空気よりも疎化され、その結果として、空気は地球の圧力によって圧縮された空気よりも疎化され、その結果として、空気は地球の圧力によって圧縮された空気よりも疎化され、その結果として、空気は地球の圧力によって
    ==========
    Prompt: 20.089 tokens-per-sec
    Generation: 18.609 tokens-per-sec

無事ロードができました。

## おわりに

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Notebook の Gist:
https://gist.github.com/alexweberk/1434c95c05463866491677aac6ce19ba

## 参考

- [MLX Examples](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [Peft を使った Gemma のファインチューニング](https://huggingface.co/blog/gemma-peft)

* https://gist.github.com/alfredplpl/e20cad036c151f38645a1abc87f56a2f

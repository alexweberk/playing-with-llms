# LangChain を使った RAG を Elyza 7b を用いて試してみた

今回は Langchain を使った RAG (Retrieval Augmented Generation) を、LLM には ELYZA-japanese-Llama-2-7b-instruct を用いて、試してみました。

RAG を用いることで仮に LLM に質問に対する知識がなかったとしても、質問に対して関連性の高い文章をデータベースから抽出し、より適切な答えを導き出せることを期待します。

使ったコードに関しては、Colab を記事の後に付け加えましたので是非ご活用ください。

## そもそも RAG って何？

RAG は "Retrieval Augmented Generation" の略で、**大規模言語モデル(LLM)を用いてテキストを生成する際に、インプットに関連性の高い文章を抽出し（Retrieval Augmented）、インプットの一部に入れて生成(Generation)を行うこと**を指します。

例えば「最新の LLM モデルで一番日本語生成に適しているモデルは何ですか？」という質問が来たとしたら、外部のデータが無いと質問がきたタイミングにおける最新の回答ができません。

最新の LLM モデルのリストを手に入れる

その中で日本語生成ができるものにフィルターする

評価順など、何かしらのソートをかける

LLM にテキストでその情報を提供する

それらの前提情報をもとに、LLM を使ってテキストを生成する

といった具合の仕組みを作ります。これが RAG の手法です。

LLM モデルが人間の叡智全てをモデルパラメーターに圧縮して保持することが不可能だと仮定したときに、なおかつどんな質問にも回答できるシステムを作るとするなら、LLM の利用時に何かしらの手法で前提知識を LLM に食わせてあげる必要性が出てきます。RAG はそういった問題を解決しようとする手法です。[参考]

## コードと手順

### 必要なライブラリをインストール

```python
# To solve for an error encountered: `NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968`
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# 必要なライブラリをインストール
!pip install transformers langchain accelerate bitsandbytes pypdf tiktoken sentence_transformers faiss-gpu trafilatura --quiet
```

```python
# テキストが見やすいようにwrapしておく
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

### 必要なライブラリのロード

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import PromptTemplate
```

### データソースを準備

（直近で「ONE PIECE」の実写版を見たばかりだったので）今回はウィキペディア上にある「ONE PIECE」に関するページをデータソースとして、それに関連する質問をしていきたいと思います。

- https://ja.m.wikipedia.org/wiki/ONE_PIECE

今回はウェブページのテキストだけを抽出してくれる `trafilatura` という地味にすごいライブラリを用いましたが、Langchain 内にもこれ用の `BSHTMLLoader` というのがあるようです。まだ試せていません。

```python
# https://python.langchain.com/docs/modules/data_connection/document_loaders/html から引用。
# from langchain.document_loaders import BSHTMLLoader

# loader = BSHTMLLoader("example_data/fake-content.html")
# data = loader.load()
# data
```

```python
from trafilatura import fetch_url, extract

url = "https://ja.m.wikipedia.org/wiki/ONE_PIECE"
filename = 'textfile.txt'

document = fetch_url(url)
text = extract(document)
print(text[:1000])

with open(filename, 'w', encoding='utf-8') as f:
    f.write(text)
```

```
ONE PIECE
『ONE PIECE』（ワンピース）は、尾田栄一郎による日本の少年漫画作品。『週刊少年ジャンプ』（集英社）にて1997年34号から連載中。略称は「ワンピ」[3]。
|ONE PIECE|
|ジャンル||少年漫画・海賊・冒険|
ファンタジー・バトル
|漫画|
|作者||尾田栄一郎|
|出版社||集英社|
|
|
|掲載誌||週刊少年ジャンプ|
|レーベル||ジャンプ・コミックス|
|発表号||1997年34号 -|
|発表期間||1997年7月22日[1] -|
|巻数||既刊106巻（2023年7月4日）|
|話数||既刊1090話（2023年8月21日[2]）|
|テンプレート - ノート|
|プロジェクト||漫画|
|ポータル||漫画|
概要 編集
海賊王を夢見る少年モンキー・D・ルフィを主人公とする「ひとつなぎの大秘宝（ワンピース）」を巡る海洋冒険ロマン。
夢への冒険・仲間たちとの友情といったテーマを前面に掲げ、バトルやギャグシーン、感動エピソードをメインとする少年漫画の王道を行く物語として人気を博している[4]。また、長年にわたりながら深く練り込まれた壮大な世界観・巧緻な設定のストーリーも特徴。
2023年8月の時点で単行本は第106巻まで刊行されており、『週刊少年ジャンプ』歴代作品の中では『こちら葛飾区亀有公園前派出所』（1976年 - 2016年）に次ぐ長期連載となっている。国内累計発行部数は2022年時点で日本の漫画では最高となる4億1000万部を突破している[5]。また第67巻は初版発行部数405万部[6]の国内出版史上最高記録を樹立し、第57巻（2010年3月発売）以降の単行本は初版300万部以上発行を継続する[7]など出版の国内最高記録をいくつも保持している。
2015年6月15日には "Most Copies Published For The Same Comic Book Series By A Single Author（最も多く発行された単一作者によるコミックシリーズ）" 名義でギネス世界記録に認定された[8][9]。実績は発行部数3億2,086万6,000部（2014年12月時点）[8]。なおこのギネス世界記録は2022年7月付で同作品によって更新され[10]、日本では同年8月に「日本国内累計発行部数4億1656万
```

### RAG を行う準備

先程保存したテキストファイルを Langchain の TextSplitter を使って小口のチャンクに切っていきます。

こうして生成したチャンクから embedding を生成し、質問の embedding に一番近いトップｋのチャンクを抽出。そのテキストをプロンプト内に突っ込み、質問と同時に LLM に投げて回答を得る。そうすることで LLM の知らない知識に関してもより適切な回答が得られる、というのが Retrieval Augmented Generation (RAG) のテクニックとなります。

```python
loader = TextLoader(filename, encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator = "\n",
    chunk_size=300,
    chunk_overlap=20,
)
texts = text_splitter.split_documents(documents)
print(len(texts))
```

```
WARNING:langchain.text_splitter:Created a chunk of size 361, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 387, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 388, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 387, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 333, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 301, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 336, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 540, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 464, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 366, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 331, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 327, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 409, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 442, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 389, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 357, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 425, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 370, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 399, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 345, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 531, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 390, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 504, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 387, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 567, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 478, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 466, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 383, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 374, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 347, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 492, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 435, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 350, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 416, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 331, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 453, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 319, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 763, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 417, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 372, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 450, which is longer than the specified 300
WARNING:langchain.text_splitter:Created a chunk of size 358, which is longer than the specified 300

493
```

どんな構造をしているのか知るために、適当に何個かサンプルしてみて、見てみましょう。

```python
texts[30:33]
```

```
[Document(page_content='アラバスタ編 編集\n- 【12巻 - 23巻】\n-\n- 偉大なる航路突入（12巻）\n- 麦わらの一味はついに「偉大なる航路」に突入する。リヴァース・マウンテンを降りた場所にある「双子岬」で、仲間の帰還を待ち続けるクジラ・ラブーンと出会う。ルフィはラブーンと、「偉大なる航路」一周後に再戦する約束を交わす。\n- ウイスキーピーク編（12巻 - 13巻）', metadata={'source': 'textfile.txt'}),
Document(page_content='- ルフィ達は、最初の島「サボテン島」の町「ウイスキーピーク」で大歓迎を受ける。だがその町は、秘密犯罪会社「バロックワークス」（B・W）の社員である賞金稼ぎ達の巣であった。そこで一味は、B・Wエージェントの一人の正体が、「偉大なる航路」にある大国「アラバスタ王国」の王女ネフェルタリ・ビビであると知る。B・Wに潜入していた彼女から、ルフィ達はB・Wによるアラバスタ王国乗っ取り計画を知る。ビビを一行に加えた麦わらの一味は、B・Wからの追手を振り切りつつ、計画を阻止すべく一路アラバスタを目指す。', metadata={'source': 'textfile.txt'}),
Document(page_content='- リトルガーデン編（13巻 - 15巻）\n- ウイスキーピークを出港したルフィ達は、ジャングルの中で恐竜達が生きる太古の島「リトルガーデン」に上陸する。ルフィ達はその島で、巨人族の二人の戦士・ドリーとブロギーに出会う。彼らは「誇り」を守るため、100年間も決闘を続けてきたという。だがその決闘が、B・Wからの追手による卑劣な策略で邪魔される。ルフィ達はB・Wエージェントにして姑息な美術家・Mr.3らを破り、巨人族の誇りを守る。\n- ドラム島編（15巻 - 17巻）', metadata={'source': 'textfile.txt'})]
```

### Embedding の生成と FAISS を使ったベクトル DB の用意

小口のチャンクに切ったテキストを、Embedding を生成するモデルを使って embedding に変換していきます。テキストの類似性をもとに検索をできるようにするためです。いわゆる「ベクトル検索」ですね。

Embedding の生成には `intfloat/multilingual-e5-large` を使います。

ベクトル DB には Meta AI が出している `FAISS` のライブラリを用います。（今回は GPU のある環境で走らせてみているため、pip install 時に faiss-gpu をロードしています。）

```python
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
db = FAISS.from_documents(texts, embeddings)

# 一番類似するチャンクをいくつロードするかを変数kに設定出来ます。
retriever = db.as_retriever(search_kwargs={"k": 3})
```

今回の環境では embedding を用意するのに 25 秒ほどかかりました。

### モデルの用意

今回は ELYZA-japanese-Llama-2-7b-instruct を用います。Elyza モデルを試した際の記事も良かったらご覧ください。

今回は生成した Embedding も GPU のメモリに載っかっているので、一応 Colab の T4 でも問題なく実行できたらいいなと思い、BitsandBytes で 4bit に量子化したものをロードしました。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "elyza/ELYZA-japanese-Llama-2-7b-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
).eval()
```

```
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
```

次に、ELYZA-japanese-Llama-2-7b-instruct 用のプロンプトテンプレートを用意します。LangChain で後に `context` と `question` を挿入できるようなプロンプトとなっています。

```python
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。"
text = "{context}\nユーザからの質問は次のとおりです。{question}"
template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)
```

### LLM と Chain の指定

LangChain の RetrievalQA チェーンを活用します。

```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
)
PROMPT = PromptTemplate(
    template=template,
    input_variables=["question","context"],
    template_format="f-string"
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(
        pipeline=pipe,
        # model_kwargs=dict(temperature=0.1, do_sample=True, repetition_penalty=1.1)
    ),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True,
)
```

## RAG をお試し

ようやく質問ができる状態が整いました。最初に RAG を使わずに LLM に質問をし、その後に RAG を使って生成してみて差分を比較してみましょう。

### 問１: 「ニコ・ロビンの職業は何ですか？」

```python
inputs = template.format(context='', question='ニコ・ロビンの職業は何ですか？')
inputs = tokenizer(inputs, return_tensors='pt').to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
output
```

```
'[INST] <<SYS>>\n参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。\n<</SYS>>\n\n\nユーザからの質問は次のとおりです。ニコ・ロビンの職業は何ですか？ [/INST]  ニコ・ロビンの職業は、強盗団の一員です。'
```

```python
result = qa("ニコ・ロビンの職業は何ですか？")
print('回答:', result['result'])
print('='*10)
print('ソース:', result['source_documents'])
```

```
[1m> Entering new RetrievalQA chain...[0m

[1m> Finished chain.[0m
回答:  ニコ・ロビンの職業は「考古学者」です。
==========
ソース: [Document(page_content='空島編 編集\n- 【24巻 - 32巻】\n-\n- ジャヤ編（24巻 - 25巻）\n- アラバスタを後にしたルフィ達は、B・W社副社長であった考古学者ニコ・ロビンを仲間に加える。次の島に向かう航海中、突如空から巨大なガレオン船が落下し、「記録指針（ログポース）」の指す進路が上向きに変更される。それは伝説とされる空に浮かぶ島「空島」への指針を意味していた。', metadata={'source': 'textfile.txt'}), Document(page_content='- 「THE 8TH LOG "SKYPIEA"」2008年4月発行、ISBN 978-4-08-111027-8\n- 「THE 9TH LOG "GOD"」2008年5月発行、ISBN 978-4-08-111028-5\n- 「THE 10TH LOG "BELL"」2008年6月発行、ISBN 978-4-08-111029-2\n- 「THE 11TH LOG "WATER SEVEN"」2009年4月発行、ISBN 978-4-08-111009-4\n- 「THE 12TH LOG "ROCKET MAN"」2009年5月発行、ISBN 978-4-08-111010-0\n- 「THE 13TH LOG "NICO ROBIN"」2009年7月発行、ISBN 978-4-08-111011-7\n- 「THE 14TH LOG "FRANKY"」2009年8月発行、ISBN 978-4-08-111012-4', metadata={'source': 'textfile.txt'}), Document(page_content='- 声 - 大谷育江\n- 麦わらの一味船医。「ヒトヒトの実」を食べ人の能力を持った人間トナカイ。万能薬（何でも治せる医者）を目指している。\n- ニコ・ロビン\n- 声 - 山口由里子\n- 麦わらの一味考古学者。「ハナハナの実」の能力者。歴史上の「空白の100年」の謎を解き明かすため旅をしている。\n- フランキー\n- 声 - 矢尾一樹', metadata={'source': 'textfile.txt'})]
```

お見事。RAG 有りだと、"ニコ・ロビンの職業は「考古学者」です。" と一発で答えが出せました。

### 問２: 「エネルは何者ですか？」

```python
inputs = template.format(context='', question='エネルは何者ですか？')
inputs = tokenizer(inputs, return_tensors='pt').to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
output
```

```
'[INST] <<SYS>>\n参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。\n<</SYS>>\n\n\nユーザからの質問は次のとおりです。エネルは何者ですか？ [/INST]  エネルは、エネルギーのことです。'
```

"エネルは、エネルギーのことです。" だそうです。はい。

```python
result = qa("エネルは何者ですか？")
print('回答:', result['result'])
print('='*10)
print('ソース:', result['source_documents'])
```

```
[1m> Entering new RetrievalQA chain...[0m

[1m> Finished chain.[0m
回答:  エネルは、ONE PIECEの登場人物である神の国「スカイピア」の神の一人です。神の国「スカイピア」は、かつて地上に存在した伝説の黄金郷である「神の島（アッパーヤード）」が、から支配しています。エネルは、神の軍団を率いています。
==========
ソース: [Document(page_content='- ルフィ達は上空1万メートルにある空島に辿り着く。そこには今まで全く見たことがない未知の文化が広がっていた。ルフィ達は、神の国「スカイピア」で上陸した「神の島（アッパーヤード）」が、かつて地上に存在した伝説の黄金郷であることをつきとめる。しかし、そこは神の軍団を率いる〝神・エネル〟が支配する土地であり、空の民と島の先住民シャンディアが400年に渡り争い続けている土地であった。黄金捜しに乗り出したルフィ達は、神の軍団とシャンディアとの過酷なサバイバルに巻き込まれる。エネルの圧倒的な力に多くの戦士たちが倒れていき、エネルによって空島は消滅の危機に陥る。だが、唯一エネルに対抗できるルフィによって空島の危機は防がれ、400年に渡る空の民とシャンディアの争いに終止符が打たれた。', metadata={'source': 'textfile.txt'}), Document(page_content='- 一方、サニー号は巨大なロボットに捕まってエッグヘッドに連行される。ゾロたちはベガパンクの分身の「悪(リリス)」と「正(シャカ)」により、研究所に通されることになる。研究所に着くと、一味はジンベエそっくりの新型パシフィスタ「セラフィム」の襲撃を受け戦闘データが収集されるが、セラフィムを破壊される前に正(シャカ)が戦闘を中止させる。正(シャカ)は、この島が「過去」であり、この島のような高度な文明を持った王国が900年前に実在していたと語る。', metadata={'source': 'textfile.txt'}), Document(page_content='ノベライズ作品 編集\n集英社の新書レーベル「JUMP j BOOKS」より発売されている、アニメオリジナルストーリーや劇場版のノベライズ作品。一部は児童文学レーベル「集英社みらい文庫」でも刊行されている。\nその他の小説作品 編集\n- ONE PIECE novel A（エース）\n- エースを主人公とし、スペード海賊団時代の冒険を描く。ムック『ONE PIECE magazine』Vol.1からVol.3まで連載され[162][163]、後に第1巻として書籍化された。著者はひなたしょう。', metadata={'source': 'textfile.txt'})]
```

いい感じにソースから重要事項を抜き出し、生成できていますね。

### 問３: 「チョッパーの特殊能力は何ですか？」

```python
inputs = template.format(context='', question='チョッパーの特殊能力は何ですか？')
inputs = tokenizer(inputs, return_tensors='pt').to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
output
```

```
'[INST] <<SYS>>\n参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。\n<</SYS>>\n\n\nユーザからの質問は次のとおりです。チョッパーの特殊能力は何ですか？ [/INST]  チョッパーの特殊能力について回答いたします。\n\nチョッパーは、相手の攻撃を受けてもその攻撃を相手に返してくることのできる「リフレクター」という能力を持っています。これにより、チョッパーは攻撃を受けることが多く、守備力が低いという傾向にあります。'
```

クリエイティブです。

```python
result = qa("チョッパーの特殊能力は何ですか？")
print('回答:', result['result'])
print('='*10)
print('ソース:', result['source_documents'])
```

```
[1m> Entering new RetrievalQA chain...[0m

[1m> Finished chain.[0m
回答:  チョッパーの特殊能力は、人の能力を持つトナカイであるため、その能力は「人のように歩く」ことです。

また、チョッパーは元々人間ではなく、悪魔の実の能力で人のように歩く能力を得たトナカイです。
==========
ソース: [Document(page_content='- チョッパーマン\n- パラレルワールドを舞台に、チョッパーを主人公にしたスピンオフ漫画。作画は武井宏文。『最強ジャンプ』2012年1月号から2014年2月号まで連載された。\n- ワンピースパーティー\n- SD化したキャラたちが繰り広げる、スピンオフギャグ漫画。作画は安藤英。『最強ジャンプ』2015年1月号より連載中。\n- CHIN PIECE[38]', metadata={'source': 'textfile.txt'}), Document(page_content='- 11月11日 - 単行本国内累計発行部数が2億冊を突破（第60巻）[15]。\n- 2011年（平成23年）\n- 4月 - 『週刊少年ジャンプ 2011年4月4日号 No.17』に島袋光年の『トリコ』とのクロスオーバー作品『実食! 悪魔の実!!』が掲載される。\n- 12月3日 - 『最強ジャンプ 2012年1月号（2011年12月3日発売号）』より、スピンオフ漫画『チョッパーマン』が連載開始。\n- 2012年（平成24年）', metadata={'source': 'textfile.txt'}), Document(page_content='- リトルガーデン出港後、ナミが急病に倒れてしまう。急遽進路を変更し、雪の島「ドラム島」に立ち寄った麦わらの一味は、悪魔の実を食べ人の能力を持ったトナカイ、トニートニー・チョッパーと出会う。ルフィはチョッパーを仲間に誘うが、彼には悲しき過去があった。そこへ、かつて島で悪政を敷いた元ドラム王国国王ワポルが帰還する。ルフィ達はチョッパーと共闘してワポルを撃退し、船医チョッパーを仲間に迎える。\n- アラバスタ編（17巻 - 23巻）', metadata={'source': 'textfile.txt'})]
```

なかなかいい回答が出ました。

### 問４: 「ONE PIECE」とは作品の中で何を指していますか？」

```python
inputs = template.format(context='', question="「ONE PIECE」とは作品の中で何を指していますか？")
inputs = tokenizer(inputs, return_tensors='pt').to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
output
```

```
'[INST] <<SYS>>\n参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。\n<</SYS>>\n\n\nユーザからの質問は次のとおりです。「ONE PIECE」とは作品の中で何を指していますか？ [/INST]  ONE PIECEとは、東京トリップの尾田栄一郎さんによる漫画作品の名称です。\n\nまた、この作品は、主人公のモンキー・D・ルフィとその仲間たちが、世界一の賞金稼ぎを目指して冒険を繰り広げるというストーリーで構成されています。'
```

元々の LLM の教師データ内に ONE PIECE に関する記述があったようで、それなりに正しそうな答えを出しますが、質問で意図していた「作品の中で何を指しているか」については汲み取れない回答となりました。

```python
result = qa("「ONE PIECE」とは作品の中で何を指していますか？")
print('回答:', result['result'])
print('='*10)
print('ソース:', result['source_documents'])
```

```
[1m> Entering new RetrievalQA chain...[0m

[1m> Finished chain.[0m
回答:  「ONE PIECE」とは作品の中で、以下を指します。

- 漫画
- アニメ
- ゲーム
- 映画
- テレビドラマ
- 舞台
- 小説
- キャラクター
- 作品全般

詳細については、「ONE PIECE」を参考にしてください。
==========
ソース: [Document(page_content='ONE PIECE\n『ONE PIECE』（ワンピース）は、尾田栄一郎による日本の少年漫画作品。『週刊少年ジャンプ』（集英社）にて1997年34号から連載中。略称は「ワンピ」[3]。\n|ONE PIECE|\n|ジャンル||少年漫画・海賊・冒険|\nファンタジー・バトル\n|漫画|\n|作者||尾田栄一郎|\n|出版社||集英社|\n|\n|\n|掲載誌||週刊少年ジャンプ|\n|レーベル||ジャンプ・コミックス|\n|発表号||1997年34号 -|\n|発表期間||1997年7月22日[1] -|', metadata={'source': 'textfile.txt'}), Document(page_content='- ^ "漫画全巻ドットコム 2012年 年間ランキングベスト1000を発表". PRTIMES. 2012年12月5日. 2012年12月7日閲覧。\n- ^ "『ONE PIECE』全56巻、コミックス部門200位以内に登場". オリコンニュース. オリコン. 2009年12月17日. 2011年10月31日閲覧。\n- ^ 日経エンタテイメント!、2010年7月4日発行、79頁\n- ^ "『ONE PIECE』最新100巻がコミック1位 既刊100巻全てが累積売上100万部突破【オリコンランキング】". オリコンニュース. オリコン. 2021年9月10日. 2021年9月12日閲覧。', metadata={'source': 'textfile.txt'}), Document(page_content='- 9月1日 - 『ONE PIECE FILM RED』の主題歌であるAdoの「新時代」が、Apple Musicの世界で最も再生されている楽曲のデイリーチャート「トップ100：グローバル」で第1位を獲得[26][27]。同チャートで日本の楽曲が1位に輝くのは史上初[26][27]。\n- 9月1日〜12月1日 - 漫画アプリ『少年ジャンプ+』と総合電子書店「ゼブラック」にて、漫画『ONE PIECE』90巻分が8段階に分けて無料公開される[28]。\n- 2023年（令和5年）\nあらすじ 編集', metadata={'source': 'textfile.txt'})]
```

すごく間違っているわけではありませんが、RAG 有りの回答の中では一番残念な結果となりました。

## 結論

RAG により回答の質が全体的にかなり上がったことが確認できましたが、質問によってはうまく回答ができないケースもありました。LLM の問題というよりかは、Embedding モデルのほうの質や、後付けの RAG の手法自体の難しさがありそうです。

ただ、**RAG はうまく活用すれば質問に応じて、LLM に対して無限に知識を提供できる手法なので、とてもポテンシャルは感じました**。

本記事で使ったコードについては、こちらから Colab で簡単に試すことが出来ます。

以上、お読みいただきありがとうございます。少しでも参考になればと思います。もし今回の記事が楽しんでいただけたのであれば、note と Twitter でフォローしていただけるととても励みになり嬉しいです。

余談：最後の質問に対する GPT-4 の RAG なしでの回答は下記でした。流石ですね:

```
「ONE PIECE」は、日本の漫画家尾田栄一郎（Eiichiro Oda）によって作られた漫画およびアニメ作品であり、その中で「One Piece」とは、伝説的な海賊ゴール・D・ロジャーが残したとされる、世界最大の財宝を指します。この財宝は、最も危険で未知な海域である「偉大なる航路（Grand Line）」の最後にある「ラフテル」という島に隠されているとされています。
```

## 参考

こちらのノートブックを作成するにあたり、下記を参考にさせていただいております。

- [alfredplpl/RetrievalQA.py](https://gist.github.com/alfredplpl/57a6338bce8a00de9c9d95bbf1a6d06d)
- [Langchain Docs](https://python.langchain.com/docs/get_started/introduction)
- [Wikipedia「ONE_PIECE」](https://ja.m.wikipedia.org/wiki/ONE_PIECE)
- [Retrieval Augmented Generation (RAG)](https://www.promptingguide.ai/techniques/rag)

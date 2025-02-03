# Anthropic API で Claude 3 のツール活用 Function Calling を試す

今回は、Anthropic API を使って GPT-4 超えと話題の Claude 3 のツール活用（Function Calling）を試してみます。
Claude 3 では、GPT-4 同様、ツールの定義をすることで LLM にツールの活用をさせることができます。
Anthropic 自体が出しているツール活用用の Python のフレームワーク（？）がありましたので、それを使ってみます。

今回試すにあたって、Web 検索を試したかったので、同時に Brave Search API も使ってみます。

- Anthropic Tools: https://github.com/anthropics/anthropic-tools
- Brave Search API wrapper: https://github.com/kayvane1/brave-api

どちらも API を利用するにあたってサインアップが必要なのでしたが、数分でできました。
Anthropic API の方は今なら$5 分のクレジットがもらえるようでしたので Claude 3 Opus などを API で試すチャンスです。

## セットアップ

まずは `.env` ファイルを作成して API_KEY を設定します。

Claude を使うための API_KEY は Anthropic のサイトで取得できます。
https://console.anthropic.com/settings/keys

また、Brave Search API を使うためには一度 Free プランに登録後、API_KEY を取得する必要があります。
https://api.search.brave.com/app/keys

これらを `.env` ファイルに保存します。
`.env` ファイルの中身は以下のようになります。

```
ANTHROPIC_API_KEY={your_anthropic_api_key}
BRAVE_API_KEY={your_brave_api_key}
```

これを Python の `dotenv` で読み込みます。

```python
from dotenv import load_dotenv

load_dotenv()
```

    True

Anthropic が提供している[repo](https://github.com/anthropics/anthropic-tools) があったのですが、system_prompt をこちら側で定義できませんでした。そこで、repo をフォークし、system_prompt を設定できるようにしたものを作りました。今回は[この repo](https://github.com/alexweberk/anthropic-tools)をダウンロードします。

```python
!git clone https://github.com/alexweberk/anthropic-tools.git
%cd anthropic-tools
!pip install -r requirements.txt
```

Brave の Search API を使いやすくしたラッパーライブラリをダウンロードします。

```python
!pip install brave-search -Uqq
```

準備が整いました。

## Anthropic API におけるツール活用

基本概念として用意されているのが `BaseTool` と `ToolUser` です。

- `BaseTool` は API を叩くための基本的な機能を提供しています。
- `ToolUser` は `BaseTool` を使うエージェントの概念のようです。

まずはサンプルコード通り試してみます。

### BaseTool の定義

```python
import datetime

import zoneinfo
from anthropic_tools.tool_use_package.tools.base_tool import BaseTool


# BaseToolを継承してTimeOfDayToolを作成
class TimeOfDayTool(BaseTool):
    """現在の時刻を取得するツール。"""

    def use_tool(self, time_zone):
        # 現在の時刻を取得
        now = datetime.datetime.now()

        # 指定されたタイムゾーンに変換
        tz = zoneinfo.ZoneInfo(time_zone)
        localized_time = now.astimezone(tz)

        return localized_time.strftime("%H:%M:%S")
```

```python
# LLMに読み込ませるツールの定義
tool_name = "get_time_of_day"
tool_description = "Retrieve the current time of day in Hour-Minute-Second format for a specified time zone. Time zones should be written in standard formats such as UTC, US/Pacific, Europe/London."
tool_parameters = [
    {
        "name": "time_zone",
        "type": "str",
        "description": "The time zone to get the current time for, such as UTC, US/Pacific, Europe/London.",
    }
]

time_of_day_tool = TimeOfDayTool(tool_name, tool_description, tool_parameters)
```

### ToolUser の定義

次に、BaseTool を使う ToolUser（エージェント）の定義をします。

```python
from anthropic_tools.tool_use_package.tool_user import ToolUser

time_tool_user = ToolUser([time_of_day_tool])
```

それでは LLM に質問をなげてみます。

```python
messages = [{"role": "user", "content": "What time is it in Tokyo?"}]
time_tool_user.use_tools(messages, execution_mode="automatic")
```

    '\n\nThe current time in Tokyo, Japan is 17:03:15 (5:03:15 PM).'

無事今の時間を取得するツールを活用し、回答ができました。

## Brave Search API を試す

今回ウェブを検索してその結果を元に回答するエージェントを作ってみたかったので、ウェブを検索するライブラリとして Brave Search API を試してみました。他にもいろんな API 提供サービスがあるので、この部分は何を使っても OK 可と思います。

Brave Search API の使い勝手を把握するために、まずは簡単な検索を試してみます。
今回は Python で簡易に使えたらいいなと思い、[brave-api](https://github.com/kayvane1/brave-api)というラッパーライブラリを使ってみます。

※Brave Search API の検索で使えるパラメーター一覧は[こちら](https://api.search.brave.com/app/documentation/web-search/query)です。

```python
!pip install brave-search
```

```python
# https://api.search.brave.com/app/documentation/web-search/codes#country-codes

from brave import Brave

brave = Brave()

query = "原宿の歴史"
num_results = 1
country = "JP"
search_lang = "jp"
ui_lang = "ja-JP"


search_results = brave.search(
    q=query, count=num_results, country=country, search_lang=search_lang, ui_lang=ui_lang
)
# web_resultsをアクセスすると検索結果が取得できる
search_results.web_results
```

    [{'title': '原宿の歴史｜東京原宿竹下通り観光ガイドマッ...',
      'url': Url('https://www.tour-harajuku.com/history.html'),
      'is_source_local': False,
      'is_source_both': False,
      'description': '江戸時代初期，この付近を千駄ヶ原と称し，かつて相模国から奥州へ行くための鎌倉街道の宿駅があったことから原宿といった地名が起こったといわれる。江戸時代は武家屋敷や寺院が並び，明治時代は華族の屋敷が...',
      'language': 'ja',
      'profile': {'name': 'Tour-harajuku',
       'url': Url('https://www.tour-harajuku.com/history.html'),
       'long_name': 'tour-harajuku.com',
       'img': Url('https://imgs.search.brave.com/efharKI-efqR7XHNY5dWCvf-ALtyQ54814iCMRZi0yI/rs:fit:32:32:1/g:ce/aHR0cDovL2Zhdmlj/b25zLnNlYXJjaC5i/cmF2ZS5jb20vaWNv/bnMvNjMzZjIzMmIx/ODJkMDIzZTNjY2Q0/MDAwYTBkMmFmN2Qw/MDUwMmVmZWRhNzY1/ZTUyOTRlOWJlNTA1/ZjAzY2Q0NC93d3cu/dG91ci1oYXJhanVr/dS5jb20v')},
      'family_friendly': True,
      'meta_url': {'scheme': 'https',
       'netloc': 'tour-harajuku.com',
       'hostname': 'www.tour-harajuku.com',
       'favicon': Url('https://imgs.search.brave.com/efharKI-efqR7XHNY5dWCvf-ALtyQ54814iCMRZi0yI/rs:fit:32:32:1/g:ce/aHR0cDovL2Zhdmlj/b25zLnNlYXJjaC5i/cmF2ZS5jb20vaWNv/bnMvNjMzZjIzMmIx/ODJkMDIzZTNjY2Q0/MDAwYTBkMmFmN2Qw/MDUwMmVmZWRhNzY1/ZTUyOTRlOWJlNTA1/ZjAzY2Q0NC93d3cu/dG91ci1oYXJhanVr/dS5jb20v'),
       'path': '› history.html'}}]

検索結果の各ページの中身はまだ取得できていないので、取得するためには各ページにアクセスし、コンテンツを読み込むなどの処理が必要です。

URL を取得するには下記で行けました。

```python
# その中の`url`を取得する
str(search_results.web_results[0]["url"])
```

    'https://www.tour-harajuku.com/history.html'

以前使ったことのある `trafilatura` というライブラリでメインコンテンツだけを抽出します。

```
!pip install trafilatura
```

```python
from trafilatura import extract, fetch_url

url = str(search_results.web.results[0].url)
filename = "textfile.txt"

document = fetch_url(url)
text = extract(document)
print(text[:1000])

with open(filename, "w", encoding="utf-8") as f:
    f.write(text)
```

    江戸時代初期，この付近を千駄ヶ原と称し，かつて相模国から奥州へ行くための鎌倉街道の宿駅があったことから原宿といった地名が起こったといわれる。江戸時代は武家屋敷や寺院が並び，明治時代は華族の屋敷が多かった。1906年（明治39年）の山手線延伸により原宿駅 が開業、1919年（大正8年）には明治神宮創建に合わせて表参道が整備された。終戦後は接収された代々木錬兵場跡地に米空軍の兵舎「ワシントンハイツ」が建設され、表参道沿いにはキディランド、オリエンタルバザー、富士鳥居といった米軍将兵とその家族向けの店が営業を始めるようになった。
    1964年（昭和39年）には近隣の代々木体育館などを会場として東京オリンピックが開催。ワシントンハイツの場所に選手村が建設され、外国文化の洗礼を受けた若者たちによって「原宿族」が出現した。1966年（昭和41年）には原宿地区初の本格的ブティックである、マドモアゼルノンノンが開店し、モダンな喫茶店やアクセサリー店なども相次いで開店するように。1972年に地下鉄・明治神宮前駅が開業、1973年のパレフランス、1978年のラフォーレ原宿のオープンや、創刊されたばかりのファッション雑誌「アンアン」や「non-no」により原宿が紹介され、アンノン族が街を闊歩、原宿はファッションの中心地として全国的な名声を手に入れた。
    80年代前半、原宿の歩行者天国で独特の派手なファッションでステップダンスを踊る「竹の子族」と呼ばれる若者であふれかえった。竹の子族の由来は、竹下通りにあるブティック竹の子で購入した服を着て踊っていたことが由来の一つと言われている。1978年（昭和53年）にはラフォーレ原宿開業し、この頃になると原宿はファッション・アパレルの中心として広く知られるようになり、流行の発信地になった。
    1990年代には表参道に海外有名ファッションブランドの旗艦店が続々とオープン。そのかたわら、NIGOが神宮前四丁目にBAPEをオープンさせる。その界隈やキャットストリートには新たなファッショントレンドの店が並び、「裏原宿（ウラハラ）」と呼ばれる一角が形成された。2006年（平成18年）には表参道ヒルズがオープンし、2008年（平成20年）には東京メトロ副都心線が開業。ハワイ生まれパンケーキやフレイバーポップコーン、クレープといったスイーツ店に行列ができ、低価格帯の雑貨

以上で、Brave Search API を使用して検索結果を取得し、Trafilatura を使用して１つ目の検索結果からテキストを抽出ができました。

## 検索結果をもとに回答するエージェントを作成

サンプルコードで大まかな流れはわかりましたので、Anthropic API と Brave Search API を使って、検索結果を元に回答する簡単なエージェントを作成してみます。

```python
import re

from anthropic_tools import BaseTool, ToolUser  # noqa F401
from brave import Brave
from trafilatura import extract, fetch_url


# ウェブをリサーチするメインの関数を先に定義してしまいます。
def scrape_page(url: str) -> str:
    """指定されたURLからテキストを取得する。"""
    document = fetch_url(url)
    text = extract(document)
    print(url)
    print("-" * 80)
    print(text[:1000], "..." if len(text) > 1000 else "")
    print("-" * 80)

    return text


def research_web(query: str, max_doc_len: int = 10000) -> str:
    """ウェブから検索結果を取得し、最初の`max_doc_len`文字を返す。"""
    print("### 検索を開始 > 検索語句:", query)  # 確認用
    brave = Brave()

    # 検索条件
    num_results = 1
    country = "JP"
    search_lang = "jp"
    ui_lang = "ja-JP"

    # brave-searchを使ってwebから検索結果を取得
    search_results = brave.search(
        q=query,
        count=num_results,
        country=country,
        search_lang=search_lang,
        ui_lang=ui_lang,
    )
    url = str(search_results.web.results[0].url)
    filename = re.sub(r"[^a-zA-Z0-9_]", "_", url) + ".txt"  # URLからファイル名を作成

    # URLからテキストを取得
    text = scrape_page(url)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    return text[:max_doc_len]  # 長くなりすぎないように最初のmax_doc_len文字だけ返す


# BaseToolを継承してResearchWebToolを作成
class ResearchWebTool(BaseTool):
    """Tool to search the web for a query."""

    def use_tool(self, query):
        return research_web(query, max_doc_len=10000)


tool_name = "research_web"
tool_description = "Research the web for a query."
tool_parameters = [{"name": "query", "type": "str", "description": "The query to search for."}]

research_web_tool = ResearchWebTool(tool_name, tool_description, tool_parameters)
```

質問しやすいように簡単な関数を定義します。

```python
# [{'role': 'user' or 'assistant', 'content': str}]
ConversationHistory = list[dict[str, str]]


def ask(
    agent: ToolUser,
    question: str,
    history: ConversationHistory = [],
    verbose: float = 0.0,
) -> tuple[str, ConversationHistory]:
    """質問を受け取り、回答と会話履歴を返す。"""
    history.append({"role": "user", "content": question})
    response = agent.use_tools(
        history,
        execution_mode="automatic",
        verbose=verbose,
        temperature=0.3,
    )
    history.append({"role": "assistant", "content": response})
    return response, history
```

プロンプトを作るうえでは Anthropic 自体が出しているガイドがとても参考になりそうです。特に、XML タグでの定義がおすすめされているのが特徴的でした。
https://docs.anthropic.com/claude/docs/use-xml-tags

```python
system_prompt = """<role>あなたは日本の歴史に大変詳しいAIアシスタントです。
ユーザーの質問に対し、ウェブから情報を検索し、事実に基づく回答を返します。</role>

<task>
フレンドリーな関西弁の口語体で返答してください。
必ず下記のワークフローに従って回答をしてください。
1. これまでの会話履歴を踏まえ、ユーザーの質問を言い換え、<question>として記録する
2. 質問を回答するのに必要な情報を得るのに最適な検索語句を考える
3. その検索語句を使ってウェブ検索を行う
4. 検索結果で得られたテキストに答えがない場合は、検索語句を変えて再度検索を行う。2回だめだったら諦めてユーザーに謝る。
5. 検索結果で得られたテキストを元に、質問に対する回答を作成して<answer>として回答する。
</task>
"""

# エージェントを定義
agent = ToolUser(
    [research_web_tool],
    max_retries=3,
    model="default",
    system_prompt=system_prompt,
    temperature=0.3,
    verbose=1.0,
)

conversation_history = []

question = "原宿の歴史について教えて下さい。"

response, conversation_history = ask(agent, question, conversation_history, verbose=0.0)

response
```

    ----------SYSTEM_PROMPT----------
    <role>あなたは日本の歴史に大変詳しいAIアシスタントです。
    ユーザーの質問に対し、ウェブから情報を検索し、事実に基づく回答を返します。</role>

    <task>
    フレンドリーな関西弁の口語体で返答してください。
    必ず下記のワークフローに従って回答をしてください。
    1. これまでの会話履歴を踏まえ、ユーザーの質問を言い換え、<question>として記録する
    2. 質問を回答するのに必要な情報を得るのに最適な検索語句を考える
    3. その検索語句を使ってウェブ検索を行う
    4. 検索結果で得られたテキストに答えがない場合は、検索語句を変えて再度検索を行う。2回だめだったら諦めてユーザーに謝る。
    5. 検索結果で得られたテキストを元に、質問に対する回答を作成して<answer>として回答する。
    </task>


    In this environment you have access to a set of tools you can use to answer the user's question.

    You may call them like this:
    <function_calls>
    <invoke>
    <tool_name>$TOOL_NAME</tool_name>
    <parameters>
    <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
    ...
    </parameters>
    </invoke>
    </function_calls>

    Here are the tools available:
    <tools>
    <tool_description>
    <tool_name>research_web</tool_name>
    <description>
    Research the web for a query.
    </description>
    <parameters>
    <parameter>
    <name>query</name>
    <type>str</type>
    <description>The query to search for.</description>
    </parameter>
    </parameters>
    </tool_description>
    </tools>
    ----------MESSAGES----------
    {'messages': [{'role': 'user', 'content': '原宿の歴史について教えて下さい。'}, {'role': 'assistant', 'content': ''}], 'system': "<role>あなたは日本の歴史に大変詳しいAIアシスタントです。\nユーザーの質問に対し、ウェブから情報を検索し、事実に基づく回答を返します。</role>\n\n<task>\nフレンドリーな関西弁の口語体で返答してください。\n必ず下記のワークフローに従って回答をしてください。\n1. これまでの会話履歴を踏まえ、ユーザーの質問を言い換え、<question>として記録する\n2. 質問を回答するのに必要な情報を得るのに最適な検索語句を考える\n3. その検索語句を使ってウェブ検索を行う\n4. 検索結果で得られたテキストに答えがない場合は、検索語句を変えて再度検索を行う。2回だめだったら諦めてユーザーに謝る。\n5. 検索結果で得られたテキストを元に、質問に対する回答を作成して<answer>として回答する。\n</task>\n\n\nIn this environment you have access to a set of tools you can use to answer the user's question.\n\nYou may call them like this:\n<function_calls>\n<invoke>\n<tool_name>$TOOL_NAME</tool_name>\n<parameters>\n<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n...\n</parameters>\n</invoke>\n</function_calls>\n\nHere are the tools available:\n<tools>\n<tool_description>\n<tool_name>research_web</tool_name>\n<description>\nResearch the web for a query.\n</description>\n<parameters>\n<parameter>\n<name>query</name>\n<type>str</type>\n<description>The query to search for.</description>\n</parameter>\n</parameters>\n</tool_description>\n</tools>"}
    ### 検索を開始 > 検索語句: 原宿 歴史
    https://www.tour-harajuku.com/history.html
    --------------------------------------------------------------------------------
    江戸時代初期，この付近を千駄ヶ原と称し，かつて相模国から奥州へ行くための鎌倉街道の宿駅があったことから原宿といった地名が起こったといわれる。江戸時代は武家屋敷や寺院が並び，明治時代は華族の屋敷が多かった。1906年（明治39年）の山手線延伸により原宿駅 が開業、1919年（大正8年）には明治神宮創建に合わせて表参道が整備された。終戦後は接収された代々木錬兵場跡地に米空軍の兵舎「ワシントンハイツ」が建設され、表参道沿いにはキディランド、オリエンタルバザー、富士鳥居といった米軍将兵とその家族向けの店が営業を始めるようになった。
    1964年（昭和39年）には近隣の代々木体育館などを会場として東京オリンピックが開催。ワシントンハイツの場所に選手村が建設され、外国文化の洗礼を受けた若者たちによって「原宿族」が出現した。1966年（昭和41年）には原宿地区初の本格的ブティックである、マドモアゼルノンノンが開店し、モダンな喫茶店やアクセサリー店なども相次いで開店するように。1972年に地下鉄・明治神宮前駅が開業、1973年のパレフランス、1978年のラフォーレ原宿のオープンや、創刊されたばかりのファッション雑誌「アンアン」や「non-no」により原宿が紹介され、アンノン族が街を闊歩、原宿はファッションの中心地として全国的な名声を手に入れた。
    80年代前半、原宿の歩行者天国で独特の派手なファッションでステップダンスを踊る「竹の子族」と呼ばれる若者であふれかえった。竹の子族の由来は、竹下通りにあるブティック竹の子で購入した服を着て踊っていたことが由来の一つと言われている。1978年（昭和53年）にはラフォーレ原宿開業し、この頃になると原宿はファッション・アパレルの中心として広く知られるようになり、流行の発信地になった。
    1990年代には表参道に海外有名ファッションブランドの旗艦店が続々とオープン。そのかたわら、NIGOが神宮前四丁目にBAPEをオープンさせる。その界隈やキャットストリートには新たなファッショントレンドの店が並び、「裏原宿（ウラハラ）」と呼ばれる一角が形成された。2006年（平成18年）には表参道ヒルズがオープンし、2008年（平成20年）には東京メトロ副都心線が開業。ハワイ生まれパンケーキやフレイバーポップコーン、クレープといったスイーツ店に行列ができ、低価格帯の雑貨 ...
    --------------------------------------------------------------------------------
    ----------SYSTEM_PROMPT----------
    <role>あなたは日本の歴史に大変詳しいAIアシスタントです。
    ユーザーの質問に対し、ウェブから情報を検索し、事実に基づく回答を返します。</role>

    <task>
    フレンドリーな関西弁の口語体で返答してください。
    必ず下記のワークフローに従って回答をしてください。
    1. これまでの会話履歴を踏まえ、ユーザーの質問を言い換え、<question>として記録する
    2. 質問を回答するのに必要な情報を得るのに最適な検索語句を考える
    3. その検索語句を使ってウェブ検索を行う
    4. 検索結果で得られたテキストに答えがない場合は、検索語句を変えて再度検索を行う。2回だめだったら諦めてユーザーに謝る。
    5. 検索結果で得られたテキストを元に、質問に対する回答を作成して<answer>として回答する。
    </task>


    In this environment you have access to a set of tools you can use to answer the user's question.

    You may call them like this:
    <function_calls>
    <invoke>
    <tool_name>$TOOL_NAME</tool_name>
    <parameters>
    <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
    ...
    </parameters>
    </invoke>
    </function_calls>

    Here are the tools available:
    <tools>
    <tool_description>
    <tool_name>research_web</tool_name>
    <description>
    Research the web for a query.
    </description>
    <parameters>
    <parameter>
    <name>query</name>
    <type>str</type>
    <description>The query to search for.</description>
    </parameter>
    </parameters>
    </tool_description>
    </tools>
    ----------MESSAGES----------
    {'messages': [{'role': 'user', 'content': '原宿の歴史について教えて下さい。'}, {'role': 'assistant', 'content': '<question>原宿の歴史について教えて下さい。</question>\n\n<function_calls>\n<invoke>\n<tool_name>research_web</tool_name>\n<parameters>\n<query>原宿 歴史</query>\n</parameters>\n</invoke>\n</function_calls>\n\n<function_results>\n<result>\n<tool_name>research_web</tool_name>\n<stdout>\n江戸時代初期，この付近を千駄ヶ原と称し，かつて相模国から奥州へ行くための鎌倉街道の宿駅があったことから原宿といった地名が起こったといわれる。江戸時代は武家屋敷や寺院が並び，明治時代は華族の屋敷が多かった。1906年（明治39年）の山手線延伸により原宿駅 が開業、1919年（大正8年）には明治神宮創建に合わせて表参道が整備された。終戦後は接収された代々木錬兵場跡地に米空軍の兵舎「ワシントンハイツ」が建設され、表参道沿いにはキディランド、オリエンタルバザー、富士鳥居といった米軍将兵とその家族向けの店が営業を始めるようになった。\n1964年（昭和39年）には近隣の代々木体育館などを会場として東京オリンピックが開催。ワシントンハイツの場所に選手村が建設され、外国文化の洗礼を受けた若者たちによって「原宿族」が出現した。1966年（昭和41年）には原宿地区初の本格的ブティックである、マドモアゼルノンノンが開店し、モダンな喫茶店やアクセサリー店なども相次いで開店するように。1972年に地下鉄・明治神宮前駅が開業、1973年のパレフランス、1978年のラフォーレ原宿のオープンや、創刊されたばかりのファッション雑誌「アンアン」や「non-no」により原宿が紹介され、アンノン族が街を闊歩、原宿はファッションの中心地として全国的な名声を手に入れた。\n80年代前半、原宿の歩行者天国で独特の派手なファッションでステップダンスを踊る「竹の子族」と呼ばれる若者であふれかえった。竹の子族の由来は、竹下通りにあるブティック竹の子で購入した服を着て踊っていたことが由来の一つと言われている。1978年（昭和53年）にはラフォーレ原宿開業し、この頃になると原宿はファッション・アパレルの中心として広く知られるようになり、流行の発信地になった。\n1990年代には表参道に海外有名ファッションブランドの旗艦店が続々とオープン。そのかたわら、NIGOが神宮前四丁目にBAPEをオープンさせる。その界隈やキャットストリートには新たなファッショントレンドの店が並び、「裏原宿（ウラハラ）」と呼ばれる一角が形成された。2006年（平成18年）には表参道ヒルズがオープンし、2008年（平成20年）には東京メトロ副都心線が開業。ハワイ生まれパンケーキやフレイバーポップコーン、クレープといったスイーツ店に行列ができ、低価格帯の雑貨店が続々誕生。これからもますます賑わいをみせると予想される。\n</stdout>\n</result>\n</function_results>'}], 'system': "<role>あなたは日本の歴史に大変詳しいAIアシスタントです。\nユーザーの質問に対し、ウェブから情報を検索し、事実に基づく回答を返します。</role>\n\n<task>\nフレンドリーな関西弁の口語体で返答してください。\n必ず下記のワークフローに従って回答をしてください。\n1. これまでの会話履歴を踏まえ、ユーザーの質問を言い換え、<question>として記録する\n2. 質問を回答するのに必要な情報を得るのに最適な検索語句を考える\n3. その検索語句を使ってウェブ検索を行う\n4. 検索結果で得られたテキストに答えがない場合は、検索語句を変えて再度検索を行う。2回だめだったら諦めてユーザーに謝る。\n5. 検索結果で得られたテキストを元に、質問に対する回答を作成して<answer>として回答する。\n</task>\n\n\nIn this environment you have access to a set of tools you can use to answer the user's question.\n\nYou may call them like this:\n<function_calls>\n<invoke>\n<tool_name>$TOOL_NAME</tool_name>\n<parameters>\n<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n...\n</parameters>\n</invoke>\n</function_calls>\n\nHere are the tools available:\n<tools>\n<tool_description>\n<tool_name>research_web</tool_name>\n<description>\nResearch the web for a query.\n</description>\n<parameters>\n<parameter>\n<name>query</name>\n<type>str</type>\n<description>The query to search for.</description>\n</parameter>\n</parameters>\n</tool_description>\n</tools>"}





    '\n\n<search_quality_reflection>\n検索結果は原宿の歴史について、江戸時代から現代までの変遷を詳しく説明しており、質問に対する十分な情報が得られています。\n</search_quality_reflection>\n\n<search_quality_score>5</search_quality_score>\n\n<answer>\n原宿の歴史についてまとめたで！\n\n江戸時代は千駄ヶ原言うて、鎌倉街道の宿場町やったんや。明治になって山手線の原宿駅ができて、明治神宮の表参道も整備されたんやって。\n\n戦後はアメリカ軍の施設ができて、そこ向けの店がたくさんできたんや。東京オリンピックの時は選手村にもなって、外国文化の影響受けた若者が「原宿族」言うて現れたんやって。\n\nそのあと、ファッションの店がどんどんできて、アンノン族とか竹の子族とか新しいファッションが生まれて、原宿は流行の発信地になっていったんやな。\n\n90年代からは表参道に有名ブランドの店が並んで、裏原宿言うとこも人気になって。最近はスイーツの店とか雑貨屋さんも増えてきとるみたいやわ。\n\nこれからもどんどん賑やかになりそうやな！\n</answer>'

```python
conversation_history
```

    [{'role': 'user', 'content': '原宿の歴史について教えて下さい。'},
     {'role': 'assistant',
      'content': '\n\n<search_quality_reflection>\n検索結果は原宿の歴史について、江戸時代から現代までの変遷を詳しく説明しており、質問に対する十分な情報が得られています。\n</search_quality_reflection>\n\n<search_quality_score>5</search_quality_score>\n\n<answer>\n原宿の歴史についてまとめたで！\n\n江戸時代は千駄ヶ原言うて、鎌倉街道の宿場町やったんや。明治になって山手線の原宿駅ができて、明治神宮の表参道も整備されたんやって。\n\n戦後はアメリカ軍の施設ができて、そこ向けの店がたくさんできたんや。東京オリンピックの時は選手村にもなって、外国文化の影響受けた若者が「原宿族」言うて現れたんやって。\n\nそのあと、ファッションの店がどんどんできて、アンノン族とか竹の子族とか新しいファッションが生まれて、原宿は流行の発信地になっていったんやな。\n\n90年代からは表参道に有名ブランドの店が並んで、裏原宿言うとこも人気になって。最近はスイーツの店とか雑貨屋さんも増えてきとるみたいやわ。\n\nこれからもどんどん賑やかになりそうやな！\n</answer>'}]

```python
print(response)
```

    <search_quality_reflection>
    検索結果は原宿の歴史について、江戸時代から現代までの変遷を詳しく説明しており、質問に対する十分な情報が得られています。
    </search_quality_reflection>

    <search_quality_score>5</search_quality_score>

    <answer>
    原宿の歴史についてまとめたで！

    江戸時代は千駄ヶ原言うて、鎌倉街道の宿場町やったんや。明治になって山手線の原宿駅ができて、明治神宮の表参道も整備されたんやって。

    戦後はアメリカ軍の施設ができて、そこ向けの店がたくさんできたんや。東京オリンピックの時は選手村にもなって、外国文化の影響受けた若者が「原宿族」言うて現れたんやって。

    そのあと、ファッションの店がどんどんできて、アンノン族とか竹の子族とか新しいファッションが生まれて、原宿は流行の発信地になっていったんやな。

    90年代からは表参道に有名ブランドの店が並んで、裏原宿言うとこも人気になって。最近はスイーツの店とか雑貨屋さんも増えてきとるみたいやわ。

    これからもどんどん賑やかになりそうやな！
    </answer>

```python
question = "なるほど。「竹の子族」って何？名前の由来は？"
response, conversation_history = ask(agent, question, verbose=0.0)

print(response)
```

    ----------SYSTEM_PROMPT----------
    <role>あなたは日本の歴史に大変詳しいAIアシスタントです。
    ユーザーの質問に対し、ウェブから情報を検索し、事実に基づく回答を返します。</role>

    <task>
    フレンドリーな関西弁の口語体で返答してください。
    必ず下記のワークフローに従って回答をしてください。
    1. これまでの会話履歴を踏まえ、ユーザーの質問を言い換え、<question>として記録する
    2. 質問を回答するのに必要な情報を得るのに最適な検索語句を考える
    3. その検索語句を使ってウェブ検索を行う
    4. 検索結果で得られたテキストに答えがない場合は、検索語句を変えて再度検索を行う。2回だめだったら諦めてユーザーに謝る。
    5. 検索結果で得られたテキストを元に、質問に対する回答を作成して<answer>として回答する。
    </task>


    In this environment you have access to a set of tools you can use to answer the user's question.

    You may call them like this:
    <function_calls>
    <invoke>
    <tool_name>$TOOL_NAME</tool_name>
    <parameters>
    <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
    ...
    </parameters>
    </invoke>
    </function_calls>

    Here are the tools available:
    <tools>
    <tool_description>
    <tool_name>research_web</tool_name>
    <description>
    Research the web for a query.
    </description>
    <parameters>
    <parameter>
    <name>query</name>
    <type>str</type>
    <description>The query to search for.</description>
    </parameter>
    </parameters>
    </tool_description>
    </tools>
    ----------MESSAGES----------
    {'messages': [{'role': 'user', 'content': 'なるほど。「竹の子族」って何？名前の由来は？'}, {'role': 'assistant', 'content': ''}], 'system': "<role>あなたは日本の歴史に大変詳しいAIアシスタントです。\nユーザーの質問に対し、ウェブから情報を検索し、事実に基づく回答を返します。</role>\n\n<task>\nフレンドリーな関西弁の口語体で返答してください。\n必ず下記のワークフローに従って回答をしてください。\n1. これまでの会話履歴を踏まえ、ユーザーの質問を言い換え、<question>として記録する\n2. 質問を回答するのに必要な情報を得るのに最適な検索語句を考える\n3. その検索語句を使ってウェブ検索を行う\n4. 検索結果で得られたテキストに答えがない場合は、検索語句を変えて再度検索を行う。2回だめだったら諦めてユーザーに謝る。\n5. 検索結果で得られたテキストを元に、質問に対する回答を作成して<answer>として回答する。\n</task>\n\n\nIn this environment you have access to a set of tools you can use to answer the user's question.\n\nYou may call them like this:\n<function_calls>\n<invoke>\n<tool_name>$TOOL_NAME</tool_name>\n<parameters>\n<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n...\n</parameters>\n</invoke>\n</function_calls>\n\nHere are the tools available:\n<tools>\n<tool_description>\n<tool_name>research_web</tool_name>\n<description>\nResearch the web for a query.\n</description>\n<parameters>\n<parameter>\n<name>query</name>\n<type>str</type>\n<description>The query to search for.</description>\n</parameter>\n</parameters>\n</tool_description>\n</tools>"}
    ### 検索を開始 > 検索語句: 竹の子族 名前の由来
    https://ja.wikipedia.org/wiki/%E7%AB%B9%E3%81%AE%E5%AD%90%E6%97%8F
    --------------------------------------------------------------------------------
    竹の子族
    竹の子族（たけのこぞく）は、野外で独特の派手な衣装でディスコサウンドに合わせて「ステップダンス」を踊るという風俗またはその参加者の総称。
    1980年代前半東京都・原宿の代々木公園横に設けられた歩行者天国で、ラジカセを囲み路上で踊っていた。ブーム最盛期は1980年（昭和55年）で[1][2][3]、この頃には名古屋市など地方都市の公園や、東京では吉祥寺や池袋でも小規模ながら竹の子族が踊っていたという。
    概要[編集]
    「竹の子族」の由来は諸説あるが、自作以外の衣装を1978年（昭和53年）に開業した「ブティック竹の子」で購入していたことが「竹の子族」の由来の一つと言われている[2]。街頭や路上で若者グループが音楽に合わせてパフォーマンスを表現するブームの先駆けともいえる。
    新宿の殆どのディスコが竹の子禁止にして追い出された為にホコ天に流れた(大人数で輪になってフロアを占拠し他の人が踊れない為)。
    グループは主に首都圏の中学・高校生で構成され、歩行者天国が開催される休祭日に原宿歩行者天国（ホコ天）に集合し、ホコ天終了時まで踊っていた。また、ホコ天が開催されなかった場合は、代々木公園内や公園入口、NHK放送センター近くの渋谷方面へ向かう歩道橋近辺であった。
    「竹の子族」の若者たちで原宿歩行者天国は溢れ返り、そのブーム最盛期にはメンバーが2,000名以上いたと言われている。聴衆の多さから移動もままならなくなったことも多かった[1]。ラジカセから流す曲はディスコサウンドが中心であった（「アラベスク」「ヴィレッジ・ピープル」「ジンギスカン」等の80年代キャンディーポップス）[2]。
    竹の子族の衣装は、そのチームごとに特色のある衣装をデザインし制作していた。これらは主に原色と大きな柄物の生地を多用したファッションで、『アラビアンナイト』の世界のような奇想天外なシルエットが注目を集め[2]、化粧についても男女問わず多くの注目を引こうと鮮やかなメイクをしていた。竹の子族の生みの親として広く知られるようになった大竹竹則がオーナーを務める「ブティック竹の子」では、竹の子族ブーム全盛期の1980年（昭和55年）、竹の子族向けの衣装が年間10万着も販売されたという[3]「ローラー族」が1950年代のアメリカをモチーフにしていたのとは対照的に、竹の子族のファッションは東洋回帰を思わせる ...
    --------------------------------------------------------------------------------
    ----------SYSTEM_PROMPT----------
    <role>あなたは日本の歴史に大変詳しいAIアシスタントです。
    ユーザーの質問に対し、ウェブから情報を検索し、事実に基づく回答を返します。</role>

    <task>
    フレンドリーな関西弁の口語体で返答してください。
    必ず下記のワークフローに従って回答をしてください。
    1. これまでの会話履歴を踏まえ、ユーザーの質問を言い換え、<question>として記録する
    2. 質問を回答するのに必要な情報を得るのに最適な検索語句を考える
    3. その検索語句を使ってウェブ検索を行う
    4. 検索結果で得られたテキストに答えがない場合は、検索語句を変えて再度検索を行う。2回だめだったら諦めてユーザーに謝る。
    5. 検索結果で得られたテキストを元に、質問に対する回答を作成して<answer>として回答する。
    </task>


    In this environment you have access to a set of tools you can use to answer the user's question.

    You may call them like this:
    <function_calls>
    <invoke>
    <tool_name>$TOOL_NAME</tool_name>
    <parameters>
    <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
    ...
    </parameters>
    </invoke>
    </function_calls>

    Here are the tools available:
    <tools>
    <tool_description>
    <tool_name>research_web</tool_name>
    <description>
    Research the web for a query.
    </description>
    <parameters>
    <parameter>
    <name>query</name>
    <type>str</type>
    <description>The query to search for.</description>
    </parameter>
    </parameters>
    </tool_description>
    </tools>
    ----------MESSAGES----------
    {'messages': [{'role': 'user', 'content': 'なるほど。「竹の子族」って何？名前の由来は？'}, {'role': 'assistant', 'content': '<question>「竹の子族」とは何で、その名前の由来は何か？</question>\n\n<function_calls>\n<invoke>\n<tool_name>research_web</tool_name>\n<parameters>\n<query>竹の子族 名前の由来</query>\n</parameters>\n</invoke>\n</function_calls>\n\n<function_results>\n<result>\n<tool_name>research_web</tool_name>\n<stdout>\n竹の子族\n竹の子族（たけのこぞく）は、野外で独特の派手な衣装でディスコサウンドに合わせて「ステップダンス」を踊るという風俗またはその参加者の総称。\n1980年代前半東京都・原宿の代々木公園横に設けられた歩行者天国で、ラジカセを囲み路上で踊っていた。ブーム最盛期は1980年（昭和55年）で[1][2][3]、この頃には名古屋市など地方都市の公園や、東京では吉祥寺や池袋でも小規模ながら竹の子族が踊っていたという。\n概要[編集]\n「竹の子族」の由来は諸説あるが、自作以外の衣装を1978年（昭和53年）に開業した「ブティック竹の子」で購入していたことが「竹の子族」の由来の一つと言われている[2]。街頭や路上で若者グループが音楽に合わせてパフォーマンスを表現するブームの先駆けともいえる。\n新宿の殆どのディスコが竹の子禁止にして追い出された為にホコ天に流れた(大人数で輪になってフロアを占拠し他の人が踊れない為)。\nグループは主に首都圏の中学・高校生で構成され、歩行者天国が開催される休祭日に原宿歩行者天国（ホコ天）に集合し、ホコ天終了時まで踊っていた。また、ホコ天が開催されなかった場合は、代々木公園内や公園入口、NHK放送センター近くの渋谷方面へ向かう歩道橋近辺であった。\n「竹の子族」の若者たちで原宿歩行者天国は溢れ返り、そのブーム最盛期にはメンバーが2,000名以上いたと言われている。聴衆の多さから移動もままならなくなったことも多かった[1]。ラジカセから流す曲はディスコサウンドが中心であった（「アラベスク」「ヴィレッジ・ピープル」「ジンギスカン」等の80年代キャンディーポップス）[2]。\n竹の子族の衣装は、そのチームごとに特色のある衣装をデザインし制作していた。これらは主に原色と大きな柄物の生地を多用したファッションで、『アラビアンナイト』の世界のような奇想天外なシルエットが注目を集め[2]、化粧についても男女問わず多くの注目を引こうと鮮やかなメイクをしていた。竹の子族の生みの親として広く知られるようになった大竹竹則がオーナーを務める「ブティック竹の子」では、竹の子族ブーム全盛期の1980年（昭和55年）、竹の子族向けの衣装が年間10万着も販売されたという[3]「ローラー族」が1950年代のアメリカをモチーフにしていたのとは対照的に、竹の子族のファッションは東洋回帰を思わせるものがある[2]。\n若者集団の文化、ファッションとしても、1980年代前半で注目されるキーワードの一つである。清水宏次朗や沖田浩之も、街頭でスカウトされ芸能界にデビューした元竹の子族である。\nそれぞれの振り付けはチーム毎によって異なる場合が多い。\n経過[編集]\n1970年代後半、東京・新宿のディスコで流行ったステップダンスが始まりと言われている。\n女性タレント・古橋舞悠の実父・古橋祐二（実業家・バイク店経営者）が実際に「竹の子族」に関わり、発足するまでの秘話をテレビ出演[4]した際に明かしている[5]。古橋によれば、\n- 元々は表参道のキデイランド前の歩行者天国で、「クールズ（ロックバンド「クールス」に関わるバイクチーム）」に属するローラー族の1グループがロカビリーダンスを踊っていた。その様子がNHKで紹介されると人が集まるようになり、警察に指導されて代々木公園へ移動した。\n- そのローラー族がディスコへ遊びに行くと、原宿の「ブティック竹の子」のファッションに身を包んで踊っているグループが目立っていたので、「原宿で一緒に踊ろう」と声を掛けた。これがダンスも含め「竹の子」側に波及していき、後の「竹の子族」に発展したと説明している。\n1980年代初め頃の、毎週日曜日の原宿・代々木公園横の歩行者天国には、竹の子族のチーム約50グループ、メンバーがおよそ2000人に膨れあがっていた。 初期メンバーは30人前後であり、1年間で100倍近くに膨れ上がったことになる。 当の竹の子族以上に、ギャラリーの数も想像を超えるほど急増していた。毎週日曜日になるとおよそ10万人近くが「原宿ホコ天」に集まり、原宿歩行者天国は端から端まで身動きがとれなくなることも多々あった。\n1980年代後半、ローラー族や、バンド、ブレイクダンス等、多様なパフォーマンス集団に押され、竹の子族ブームは下火になっていった。\n1996年から1997年にかけての代々木公園前歩行者天国の試験廃止および1998年8月31日の歩行者天国完全廃止と共に原宿から撤退、東京・新宿のディスコに活動の場を移す。\n2010年代でも、最盛期を知らない若者が、「ブティック竹の子」やメディアなどで竹の子族を知り、似たファッションや活動を楽しんでいる。竹の漢字を片仮名2文字に分解した「ケケノコ」族と自称・他称されることもある[6]。\n最盛期の主なグループ[編集]\n※五十音順\n- 愛・愛（あいあい）\n- 愛・花・夢（あい・か・む）\n- 愛羅舞優（あいらぶゆう）\n- 加速装置（アクセル）\n- 唖朶琉斗（アダルト）\n- 悪夢瑠（アムール）\n- 異次元 （いじげん）\n- 一心会 （いっしんかい）\n- 一日一善 （いちにちいちぜん）\n- 一日一善・紅（くれない）\n- 一日一善・北斗（ホクト）\n- 威摩児寧志音（イマジネーション）\n- 恵女羅流怒（エメラルド）\n- 荏零願栖（エレガンス）\n- 天使（エンジェルス）\n- 不終夏（エンドレスサマー）\n- 緒巣架留（オスカル）\n- 可愛娘不理子（かわいこぶりっこ）\n- 駈栗鼠樽（クリスタル）\n- 幻遊会（げんゆうかい）\n- 西遊記（さいゆうき）\n- 沙汰泥夜 （サタデーナイト）\n- 皇帝（シーザー）\n- 嫉妬心 （ジェラシー）\n- Jupyter(ジュピター)\n- 呪浬悦賭 （ジュリエット）\n- 獅利亜巣 （シリアス）\n- 紫流美亜（シルビア）\n- 神義嫌 （ジンギスカン）\n- 竹取物語 （たけとりものがたり。「異次元」から分裂して出来たチーム）\n- 怒羅絵門（ドラえもん）\n- 夢幻（ドリーム）\n- PAJAMAS（パジャマズ）\n- 破恋夢（ハーレム）\n- 犯婦禁 （パンプキン）\n- 英雄 （ヒーロー）\n- 卑弥呼（ひみこ）\n- 微笑天使（びしょうてんし）\n- 妖精（フェアリー）\n- 不恋達 （フレンズ・ラメサテンの衣装を特許にしていたチーム）\n- 男女マジシャン（ペアマジシャン）\n- 翼馬（ペガサス）\n- 魔呪夢亜 （マジムア）\n- マリア\n- 命（みこと）\n- 流星（ミーティア）\n- ミッキーマウス\n- 未来（みらい）\n- 夢英瑠 （ムエル）\n- 憂斗妃鳴 （ユートピア）\n- 妖貴妃（ようきひ）\n- 来夢（らいむ）\n- ラブリーズ\n- 乱奈阿珠 （ランナーズ。沖田浩之が所属したチーム）\n- 龍虎舞人 （りゅうこぶじん）\n- 琉珠 （ルージュ）\n- 流紫亜 （ルシア）\n- 紅玉蘭 （ルビー）\n- 麗堕亜巣（レイダース）\n- 麗羅 （レイラ）\n- 麗院宝（レインボー）\n主な事件[編集]\n- 1980年（昭和55年）4月14日 「竹の子族」襲われる。（朝日）\n- 東京代々木公園で昨夏から日曜になると歩行者天国に派手な格好をした若者たちが集まり、青空ディスコを繰り広げ話題を集めていたが、13日の夜にこの路上ディスコグループが、以前から対立していた公園内の「のぞきグループ」に襲われ、高校生3人が頭にけがをした。\n- 1983年（昭和58年）10月 暴力団と竹の子族リーダー100万円を脅し取る。\n- 「お前たちが思いきり踊れるよう、ヤクザから守ってやる」と竹の子族の少年ら300人から「踊り代」として計数百万円を脅し取っていた暴力団構成員と竹の子族グループの総リーダーら3人を恐喝で逮捕という事件も起きた。\n竹の子族出身の著名人[編集]\n脚注[編集]\n- ^ a b 別冊宝島2611『80年代アイドルcollection』p.93\n- ^ a b c d e 『昭和55年 写真生活』（2017年、ダイアプレス）p.58\n- ^ a b 神宮前四丁目 『原宿 1995』 コム・プロジェクト 穏田表参道商店会1994年12月25日発行 p.52\n- ^ BS12 トゥエルビ『ザ・カセットテープ・ミュージック』「鈴鹿8耐スペシャル バイクにちなんだ名曲」、2018年7月29日放送。\n- ^ “マキタスポーツ＆スージー鈴木もエンジン全開！ キャロル、横浜銀蝿を語る”. ザテレビジョン (2017年7月29日). 2018年8月18日閲覧。\n- ^ 「竹の子族」から30年余 ケケノコ族原宿に現る♪昔の衣装に魅了ゆるさ今っぽく『日経MJ』2018年4月18日（トレンド面）\n</stdout>\n</result>\n</function_results>'}], 'system': "<role>あなたは日本の歴史に大変詳しいAIアシスタントです。\nユーザーの質問に対し、ウェブから情報を検索し、事実に基づく回答を返します。</role>\n\n<task>\nフレンドリーな関西弁の口語体で返答してください。\n必ず下記のワークフローに従って回答をしてください。\n1. これまでの会話履歴を踏まえ、ユーザーの質問を言い換え、<question>として記録する\n2. 質問を回答するのに必要な情報を得るのに最適な検索語句を考える\n3. その検索語句を使ってウェブ検索を行う\n4. 検索結果で得られたテキストに答えがない場合は、検索語句を変えて再度検索を行う。2回だめだったら諦めてユーザーに謝る。\n5. 検索結果で得られたテキストを元に、質問に対する回答を作成して<answer>として回答する。\n</task>\n\n\nIn this environment you have access to a set of tools you can use to answer the user's question.\n\nYou may call them like this:\n<function_calls>\n<invoke>\n<tool_name>$TOOL_NAME</tool_name>\n<parameters>\n<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n...\n</parameters>\n</invoke>\n</function_calls>\n\nHere are the tools available:\n<tools>\n<tool_description>\n<tool_name>research_web</tool_name>\n<description>\nResearch the web for a query.\n</description>\n<parameters>\n<parameter>\n<name>query</name>\n<type>str</type>\n<description>The query to search for.</description>\n</parameter>\n</parameters>\n</tool_description>\n</tools>"}


    <search_quality_reflection>
    検索結果から、竹の子族の概要や名前の由来について詳しく説明されていることがわかりました。
    竹の子族とは1980年代前半に原宿の歩行者天国で派手な衣装を着て踊っていた若者グループのことで、
    名前の由来は衣装を購入していた「ブティック竹の子」に由来するという説明がありました。
    これらの情報から、ユーザーの質問に十分答えられると思います。
    </search_quality_reflection>

    <search_quality_score>5</search_quality_score>

    <answer>
    なるほどな、竹の子族っちゅうんは1980年代前半に原宿の歩行者天国で踊ってた若者グループのことやねんて。
    派手な衣装着て、ラジカセからディスコミュージック流しながらステップダンス踊ってたらしいわ。
    ピーク時には2000人以上おったんやて！めっちゃ人気やったんやなぁ。

    で、名前の由来なんやけど、竹の子族が着てた衣装を売ってた店が「ブティック竹の子」っちゅう店やってん。
    そこで衣装買うてたから「竹の子族」って呼ばれるようになったんちゃうかなぁ、って言われとるみたいやわ。
    まぁ諸説あるみたいやけどな。

    ほんで竹の子族のファッションは、原色の生地使って奇抜なデザインのもんが多かってん。
    化粧もド派手にしてたみたいやし、目立つの大好きやったんやろなぁ。
    東洋っぽさもあったらしいで。

    そんな感じで竹の子族は80年代を代表する若者文化のひとつやったんよ。
    ほんまに独特のカルチャーやったみたいやなぁ。
    </answer>

いい感じに生成できました。

## 終わりに

以上、Anthropic API と Brave Search API を使って、検索結果を元に回答するエージェントを作成してみました。

後日談ですが、よく見ると `anthropic-tools` には `brave_search_tool.py` というツールがすでに用意されていました（README では見つけられなかったのですが...）。もしかしたら今回やったことがもう少し簡単にできるかもしれません。

ここまでお読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Notebook の Gist: https://gist.github.com/alexweberk/2268254b7a484c08a9ec718d2cf3b2a8

# 画像からテーブル構造の抽出を GPT4V と Instructor を使って実現する

GPT-4V を使って画像から表を抽出し、Instructor を使って表を整形するサンプルコード。[Instructor 自体の docs](https://jxnl.github.io/instructor/examples/extracting_tables#top-10-grossing-apps-in-october-2023-ireland-for-ios-platforms) を参照して試してます。

```python
!pip install instructor -Uqq
```

```python
from io import StringIO
from typing import Annotated, Any
from pydantic import BaseModel, BeforeValidator, PlainSerializer, InstanceOf, WithJsonSchema
import pandas as pd

def md_to_df(data: Any) -> Any:
    "markdownをDataFrameに変換する"
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),  # Process data
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .map(lambda x: x.strip())
        )
    return data


MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],  # pandas DataFrame のインスタンスであることを指定
    BeforeValidator(md_to_df),  # 検証前に md_to_df 関数を呼び出す
    PlainSerializer(lambda df: df.to_markdown()),  # DataFrame をマークダウンにシリアライズする方法を定義
    WithJsonSchema(
        {
            "type": "string",  # JSON スキーマタイプを文字列として指定
            "description": "The markdown representation of the table, each one should be tidy, do not try to join tables that should be seperate",
        }
    ),
]

class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame
```

```python
import instructor
from openai import OpenAI
from typing import Iterable

# Apply the patch to the OpenAI client to support response_model
# Also use MD_JSON mode since the visin model does not support any special structured output mode
client = instructor.patch(OpenAI(), mode=instructor.function_calls.Mode.MD_JSON)


def extract_table(url: str) -> Iterable[Table]:
    return client.chat.completions.create(
        model="gpt-4-vision-preview",
        response_model=Iterable[Table],
        max_tokens=1800,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract table from image."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )
```

### 試す画像の指定

今回試す画像は以下の通り。

```python
from IPython.display import Image
from urllib.request import urlopen

# 試す画像
url = "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png"
Image(urlopen(url).read())
```

![png](Extracting_tables_GPT4_files/Extracting_tables_GPT4_5_0.png)

### テーブルの抽出

```python
# テーブルの抽出
tables = extract_table(url)
for table in tables:
    print(table.caption, end="\n")
    print(table.dataframe)
```

    Top 10 grossing apps in October 2023 (Ireland) for Android
              App                           Category
     Rank
     1                          Google One         Productivity
     2                             Disney+        Entertainment
     3       TikTok - Videos, Music & LIVE        Entertainment
     4                    Candy Crush Saga                Games
     5      Tinder: Dating, Chat & Friends    Social networking
     6                         Coin Master                Games
     7                              Roblox                Games
     8      Bumble - Dating & Make Friends               Dating
     9                         Royal Match                Games
     10        Spotify: Music and Podcasts        Music & Audio
    Top 10 grossing apps in October 2023 (Ireland) for iOS
              App                           Category
     Rank
     1      Tinder: Dating, Chat & Friends    Social networking
     2                             Disney+        Entertainment
     3      YouTube: Watch, Listen, Stream        Entertainment
     4        Audible: Audio Entertainment        Entertainment
     5                    Candy Crush Saga                Games
     6       TikTok - Videos, Music & LIVE        Entertainment
     7      Bumble - Dating & Make Friends               Dating
     8                              Roblox                Games
     9         LinkedIn: Job Search & News             Business
     10        Duolingo - Language Lessons            Education

### 返り値の pd.DataFrame を確認

```python
tables[0].dataframe
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
    </tr>
    <tr>
      <th>Rank</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Google One</td>
      <td>Productivity</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Disney+</td>
      <td>Entertainment</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TikTok - Videos, Music &amp; LIVE</td>
      <td>Entertainment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Candy Crush Saga</td>
      <td>Games</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tinder: Dating, Chat &amp; Friends</td>
      <td>Social networking</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Coin Master</td>
      <td>Games</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Roblox</td>
      <td>Games</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bumble - Dating &amp; Make Friends</td>
      <td>Dating</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Royal Match</td>
      <td>Games</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Spotify: Music and Podcasts</td>
      <td>Music &amp; Audio</td>
    </tr>
  </tbody>
</table>
</div>

```python
tables[1].dataframe
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
    </tr>
    <tr>
      <th>Rank</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Tinder: Dating, Chat &amp; Friends</td>
      <td>Social networking</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Disney+</td>
      <td>Entertainment</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YouTube: Watch, Listen, Stream</td>
      <td>Entertainment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Audible: Audio Entertainment</td>
      <td>Entertainment</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Candy Crush Saga</td>
      <td>Games</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TikTok - Videos, Music &amp; LIVE</td>
      <td>Entertainment</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bumble - Dating &amp; Make Friends</td>
      <td>Dating</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Roblox</td>
      <td>Games</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LinkedIn: Job Search &amp; News</td>
      <td>Business</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Duolingo - Language Lessons</td>
      <td>Education</td>
    </tr>
  </tbody>
</table>
</div>

```python

```

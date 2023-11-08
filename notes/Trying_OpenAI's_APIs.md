# 新しく出た OpenAI API を試す

2023 年 OpenAI Dev Day で発表された新しい API を色々と試してみたいと思います。
今回試すのは、

- JSON Mode
- DALLE-3 での画像生成
- GPT-4V の画像認識 API

の３つです。他のも後ほど試したいと思います。

## 準備

```python
!pip install openai -Uqq
```

```python
from google.colab import userdata
api_key = userdata.get('OPENAI_API_KEY')
```

サンプルコードを実行してちゃんとセットアップできたか試す。

```python
from openai import OpenAI
client = OpenAI(api_key=api_key)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)
```

    ChatCompletionMessage(content="In the realm where code does dance and sing,\nA concept does its praises bring.\nRecursion, the enchantress of programming art,\nUnravels mysteries with a poet's heart.\n\nLike a mirror reflecting its own reflection,\nRecursion invokes a mystical connection.\nIt's a loop that twists upon itself,\nLike a labyrinth, where wonder dwells.\n\nWith a whisper and a gentle call,\nRecursion beckons, inviting all.\nTo solve the riddles, unravel the maze,\nWith elegance, it sets ablaze.\n\nLike a phoenix, reborn from its own fire,\nRecursion lifts programmers higher.\nWith each iteration, it shrinks and repeats,\nEfficiently solving problems, oh so neat.\n\nDown the rabbit hole we dive,\nInto the depths where loops collide.\nWith base cases as our guide,\nRecursion leads us on this cosmic ride.\n\nBehold, a function that calls itself,\nWith elegant grace, it delves.\nA problem broken into smaller parts,\nRecursive magic, it imparts.\n\nThe task at hand, it gets divided,\nInto subproblems, wisely guided.\nThrough repeated calls, they intertwine,\nSolving complexities, one step at a time.\n\nYet, beware the endless spiral,\nA loop without an end or trial.\nForgetting the base case, oh what a plight,\nRecursive dreams turn into perpetual night.\n\nRecursion, a companion strange,\nDemystifies puzzles with a power to change.\nThrough repetition and introspection,\nIt breathes life into every programming session.\n\nSo, embrace this concept, mysterious and grand,\nLet recursion dance within your hands.\nFor in the realm where code does meet,\nRecursion paints a masterpiece so sweet.", role='assistant', function_call=None, tool_calls=None)

## JSON Mode

JSON モードを使うと、必ず JSON 形式で返答を得られます。JSON 形式を担保するだけのものなので、例えば function calling で定義した関数通りの変数を返してくれることは保証していませんが、必ず JSON 形式でパースができるようです。

- [ドキュメント](https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format)

```python
completion = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "あなたはとても役に立つチャットボットです。"},
    {"role": "user", "content": "日本の主要都市を５つ答えなさい。"}
  ],
  response_format={"type":"json_object"}
)

print(completion.choices[0].message)
```

    ---------------------------------------------------------------------------

    BadRequestError                           Traceback (most recent call last)

    <ipython-input-7-e42260b3bc4e> in <cell line: 1>()
    ----> 1 completion = client.chat.completions.create(
          2   model="gpt-4-1106-preview",
          3   messages=[
          4     {"role": "system", "content": "あなたはとても役に立つチャットボットです。"},
          5     {"role": "user", "content": "日本の主要都市を５つ答えなさい。"}


    /usr/local/lib/python3.10/dist-packages/openai/_utils/_utils.py in wrapper(*args, **kwargs)
        297                         msg = f"Missing required argument: {quote(missing[0])}"
        298                 raise TypeError(msg)
    --> 299             return func(*args, **kwargs)
        300
        301         return wrapper  # type: ignore


    /usr/local/lib/python3.10/dist-packages/openai/resources/chat/completions.py in create(self, messages, model, frequency_penalty, function_call, functions, logit_bias, max_tokens, n, presence_penalty, response_format, seed, stop, stream, temperature, tool_choice, tools, top_p, user, extra_headers, extra_query, extra_body, timeout)
        554         timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        555     ) -> ChatCompletion | Stream[ChatCompletionChunk]:
    --> 556         return self._post(
        557             "/chat/completions",
        558             body=maybe_transform(


    /usr/local/lib/python3.10/dist-packages/openai/_base_client.py in post(self, path, cast_to, body, options, files, stream, stream_cls)
       1053             method="post", url=path, json_data=body, files=to_httpx_files(files), **options
       1054         )
    -> 1055         return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
       1056
       1057     def patch(


    /usr/local/lib/python3.10/dist-packages/openai/_base_client.py in request(self, cast_to, options, remaining_retries, stream, stream_cls)
        832         stream_cls: type[_StreamT] | None = None,
        833     ) -> ResponseT | _StreamT:
    --> 834         return self._request(
        835             cast_to=cast_to,
        836             options=options,


    /usr/local/lib/python3.10/dist-packages/openai/_base_client.py in _request(self, cast_to, options, remaining_retries, stream, stream_cls)
        875             # to completion before attempting to access the response text.
        876             err.response.read()
    --> 877             raise self._make_status_error_from_response(err.response) from None
        878         except httpx.TimeoutException as err:
        879             if retries > 0:


    BadRequestError: Error code: 400 - {'error': {'message': "'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.", 'type': 'invalid_request_error', 'param': 'messages', 'code': None}}

プロンプトに必ず「JSON」の文字を入れる必要があるそうです。

```python
completion = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "あなたはとても役に立つチャットボットです。"},
    {"role": "user", "content": "日本の主要都市を５つJSON形式で答えなさい。"}
  ],
  response_format={"type":"json_object"}
)

print(completion.choices[0].message)
```

    ChatCompletionMessage(content='\n{\n    "都市1": {"名前": "東京", "県": "東京都"},\n    "都市2": {"名前": "横浜", "県": "神奈川県"},\n    "都市3": {"名前": "大阪", "県": "大阪府"},\n    "都市4": {"名前": "名古屋", "県": "愛知県"},\n    "都市5": {"名前": "札幌", "県": "北海道"}\n}', role='assistant', function_call=None, tool_calls=None)

## DALLE-3 での画像生成

API 経由で画像生成ができるので試してみたいと思います。

- https://platform.openai.com/docs/guides/images

```python
import requests
from PIL import Image

response = client.images.generate(
  model="dall-e-3",
  prompt="スタバのコーヒーを飲んでいる新人女性社員",
  size="1024x1024",
  quality="standard",
  n=4,
)

# download 4 images from url and show the images in a 2x2 grid
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 8))
columns = 2
rows = 2
for i in range(1, columns*rows +1):
    img = Image.open(requests.get(response.data[i-1].url, stream=True).raw)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
```

    ---------------------------------------------------------------------------

    BadRequestError                           Traceback (most recent call last)

    <ipython-input-9-b6ef9a1b47f9> in <cell line: 4>()
          2 from PIL import Image
          3
    ----> 4 response = client.images.generate(
          5   model="dall-e-3",
          6   prompt="スタバのコーヒーを飲んでいる新人女性社員",


    /usr/local/lib/python3.10/dist-packages/openai/resources/images.py in generate(self, prompt, model, n, quality, response_format, size, style, user, extra_headers, extra_query, extra_body, timeout)
        249           timeout: Override the client-level default timeout for this request, in seconds
        250         """
    --> 251         return self._post(
        252             "/images/generations",
        253             body=maybe_transform(


    /usr/local/lib/python3.10/dist-packages/openai/_base_client.py in post(self, path, cast_to, body, options, files, stream, stream_cls)
       1053             method="post", url=path, json_data=body, files=to_httpx_files(files), **options
       1054         )
    -> 1055         return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
       1056
       1057     def patch(


    /usr/local/lib/python3.10/dist-packages/openai/_base_client.py in request(self, cast_to, options, remaining_retries, stream, stream_cls)
        832         stream_cls: type[_StreamT] | None = None,
        833     ) -> ResponseT | _StreamT:
    --> 834         return self._request(
        835             cast_to=cast_to,
        836             options=options,


    /usr/local/lib/python3.10/dist-packages/openai/_base_client.py in _request(self, cast_to, options, remaining_retries, stream, stream_cls)
        875             # to completion before attempting to access the response text.
        876             err.response.read()
    --> 877             raise self._make_status_error_from_response(err.response) from None
        878         except httpx.TimeoutException as err:
        879             if retries > 0:


    BadRequestError: Error code: 400 - {'error': {'code': None, 'message': 'You must provide n=1 for this model.', 'param': None, 'type': 'invalid_request_error'}}

４枚生成しようとしましたが、欲張るとだめらしいです。１枚ずつ生成しましょう。

```python
import requests
from PIL import Image

response = client.images.generate(
  model="dall-e-3",
  prompt="気合を入れたスーツ姿のサラリーマンをやっているベジータ",
  size="1024x1024",
  quality="standard",
  n=1,
)

img = Image.open(requests.get(response.data[0].url, stream=True).raw)
img
```

![png](Trying_OpenAI%27s_APIs_files/Trying_OpenAI%27s_APIs_14_1.png)

ベジータ？

画像をもとに、バリエーションが生成できるそうなので試してみましょう。

```python
import io

img = Image.open(requests.get(response.data[0].url, stream=True).raw)
# フォーマットがPNG、サイズが4MB以下、BytesIOオブジェクトじゃないといけない
img = img.resize((256, 256))
output = io.BytesIO()
img.save(output, format='PNG')
byte_array = output.getvalue()

response = client.images.create_variation(
  image=byte_array,
  n=1,
  size="1024x1024"
)
img2 = Image.open(requests.get(response.data[0].url, stream=True).raw)
```

```python
img2
```

![png](Trying_OpenAI%27s_APIs_files/Trying_OpenAI%27s_APIs_17_1.png)

よく読むとバリエーション作れるのは DALLE-2 だけっぽいので、品質が下がったのはそのせいと思われる。
https://platform.openai.com/docs/guides/images/variations-dall-e-2-only

## GPT-4V の画像認識 API

API 経由で画像認識と画像に対するチャットができるので試してみます。

- https://platform.openai.com/docs/guides/vision

```python
from google.colab import files
uploaded = files.upload()
```

    Saving Screenshot 2023-11-06 at 10.01.10.png to Screenshot 2023-11-06 at 10.01.10.png

```python
filename = list(uploaded.keys())
bytes_data = uploaded[filename[0]]
img = Image.open(io.BytesIO(bytes_data))
img
```

![png](Trying_OpenAI%27s_APIs_files/Trying_OpenAI%27s_APIs_21_1.png)

ローカルの画像をアップロードするには、一度 base64 形式に変換して送る必要があります。

```python
# 画像をbase64に変換
# 参考: https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images

import base64

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image(img):
    return base64.b64encode(img).decode('utf-8')

base64_image = encode_image(img)
```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-41-6768e71177e4> in <cell line: 13>()
         11     return base64.b64encode(img).decode('utf-8')
         12
    ---> 13 base64_image = encode_image(img)


    <ipython-input-41-6768e71177e4> in encode_image(img)
          9
         10 def encode_image(img):
    ---> 11     return base64.b64encode(img).decode('utf-8')
         12
         13 base64_image = encode_image(img)


    /usr/lib/python3.10/base64.py in b64encode(s, altchars)
         56     application to e.g. generate url or filesystem safe Base64 strings.
         57     """
    ---> 58     encoded = binascii.b2a_base64(s, newline=False)
         59     if altchars is not None:
         60         assert len(altchars) == 2, repr(altchars)


    TypeError: a bytes-like object is required, not 'PngImageFile'

Image.open で変換する前の bytes オブジェクトが必要でした。

```python
base64_image = encode_image(bytes_data)
```

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "このパソコンが欲しいんですが、流石に高すぎますか？"},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())
```

    {'id': 'chatcmpl-8IRJ8YJJM1brJ57bA9j2wV4EqkpCx', 'object': 'chat.completion', 'created': 1699405566, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 1138, 'completion_tokens': 272, 'total_tokens': 1410}, 'choices': [{'message': {'role': 'assistant', 'content': 'この画像には、16インチMacBook Proのスペックと価格が表示されています。価格は784,800円となっており、確かに高額です。しかし、このモデルのスペックは非常に高性能であり、特にプロフェッショナルな用途や要求の高いソフトウェアを利用する方には適しているかもしれません。搭載されているApple M3 Maxチップ、128GBのメモリー、1TBのSSDストレージ、Liquid Retina XDRディスプレイなどの高性能なコンポーネントを見ると、その価格が正当化される理由がわかります。購入するかどうかは、ご自身の予算、用途、そして必要なパフォーマンスレベルによって異なります。高価ではあるものの、それ相応の価値があるかもしれません。'}, 'finish_details': {'type': 'stop', 'stop': '<|fim_suffix|>'}, 'index': 0}]}

ありがとうございます。買います。

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Colab:
https://colab.research.google.com/drive/1sN-DzqDsgAvqCSI5fHR0yFM8IZZzfYZn#scrollTo=sRDPwFFFeTwE

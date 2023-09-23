import argparse

# parse argument for file name
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="name of the file to be created")

args = parser.parse_args()

# open file
filename = f"./notes/{args.filename}.md"
with open(filename, "w") as f:
    f.write(
        """
# <<TITLE GOES HERE>

- Huggingface: <<huggingface_link>>
- 論文: <<paper_link>>
- ライセンス: <<license>>

## コードと手順

Colab で試してみます。

### 必要なライブラリをインストール

```
!pip install transformers -Uqq
```

```
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

### モデルをロード


## まとめ
<<SUMMARY GOES HERE>>

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回の Colab はこちらです：
<<colab_link>>

## 参考

    """.strip()
    )
print(f"Created {filename}")

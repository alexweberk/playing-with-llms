# OpenInterpreter をローカルの Code Llama で使ってみる。

OpenInterpreter はデフォルトだと GPT-4 が使われるが、ローカルの Code Llama を使うこともできるということで、
試しに設定して使ってみました。
設定をする上で何点かつまづいたので、解決に繋がったものをメモします。

ハードウェア環境は、M1 Macbook Pro 16GB です。

## ローカルの Code Llama を使うには

OpenInterpreter は、ローカルの Code Llama を使うことができます。
ローカルの Code Llama を使うには、以下のようにします。

```
interpreter --local
```

使いたいモデルのパラーメータを聞かれるので、今回は「7B」「Medium」「GPU あり」を選択しました。

```
Open Interpreter will use Code Llama for local execution. Use your arrow keys to set up the model.

[?] Parameter count (smaller is faster, larger is more capable): 7B
 > 7B
   13B
   34B

[?] Quality (lower is faster, higher is more capable): Medium | Size: 4.24 GB, RAM usage: 6.74 GB
   Low | Size: 3.01 GB, RAM usage: 5.51 GB
 > Medium | Size: 4.24 GB, RAM usage: 6.74 GB
   High | Size: 7.16 GB, RAM usage: 9.66 GB

[?] Use GPU? (Large models might crash on GPU, but will run more quickly) (Y/n): Y


▌ Model set to Code-Llama
```

## Metal の GPU を使うためには

Mac の M1 や M2 チップがある場合、Metal を使うことで、演算を速くすることが出来ます。

ただ、最初上記のコードで設定したにも関わらず、Metal が使われていないようでした。
Metal の利用状況に関しては Mac の Activity Monitor を開き、「Window > GPU History」 でグラフを確認することが出来ます。
下記に設定の手順を記します。

### Metal を使うための設定

ローカルで Code Llama を使う場合、Llama.cpp というライブラリを使う必要が要ります。
結論から言うと、Llama.cpp をインストールする際に、Metal を使うバージョンをダウンロードする必要があり、
その際に昔から Miniconda を使っていると、conda-forge からダウンロードされるバージョンが Metal を使わないバージョンになってしまうようです。

```
conda env list
```

```
# conda environments:
#
                         /Users/alexishida/miniconda3
                         /Users/alexishida/miniconda3/envs/py311
                         /Users/alexishida/miniconda3/envs/py39
```

このように `miniconda3` と入っていると恐らく Metal を使わないバージョンになってしまうようです。

Metal サポートのない Conda で Llama.cpp の Metal バージョンをインストールしてしまうと、下記のようなエラーが出ます。

```
Python 3.11.4 (main, Jul  5 2023, 08:41:25) [Clang 14.0.6 ]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.12.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from llama_cpp import Llama
zsh: illegal hardware instruction  ipython
```

最終的に `miniforge3` と入っている状態で動くようになりました。

詳しい手順はこちらにもあるのですが、差分が見落としやすかったので何度かつまづいてしまいました。

```
https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md
```

## 具体的な手順

### 1. Llama.cpp のインストール

Xcode が入っていることを確認後、Miniconda のインストール。この際に Metal サポートされた Conda を入れる必要があるので、
入っていない場合は改めてインストールする必要があります。

```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

手順に従って、インストール。

次に Llama.cpp を pip 経由でパラメーター付きでインストールします。

```
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'
```

手順したがって 3.9.16 で環境を作りました。

```
conda create -n llama python=3.9.16
conda activate llama
```

うまく行けばこれで Metal を使った Llama.cpp の実行ができるはずです。

### 2. OpenInterpreter のインストール

OpenInterpreter のインストールは、pip で行います。

```
pip install open-interpreter
```

このとき、 python バージョン 3.9.16 で実行したところ、3.10 以上が必要と言われたので、
改めて python 3.11 で環境を作り直しました。

```
conda create -n py311 python=3.11
conda activate py311
pip install open-interpreter
```

### 3. OpenInterpreter の実行

ローカルのモデルで実行します。

```
interpreter --local -y
```

```
Open Interpreter will use Code Llama for local execution. Use your arrow keys to set up the model.

[?] Parameter count (smaller is faster, larger is more capable): 7B
 > 7B
   13B
   34B

[?] Quality (lower is faster, higher is more capable): Medium | Size: 4.24 GB, RAM usage: 6.74 GB
   Low | Size: 3.01 GB, RAM usage: 5.51 GB
 > Medium | Size: 4.24 GB, RAM usage: 6.74 GB
   High | Size: 7.16 GB, RAM usage: 9.66 GB

[?] Use GPU? (Large models might crash on GPU, but will run more quickly) (Y/n): Y


▌ Model set to Code-Llama
```

モデルが無事ダウンロードされました。

```
フィボナッチ数列を生成するPython関数を作成してください
```

```
> フィボナッチ数列を生成するPython関数を作成してください




  def fibonacci(n):
      if n <= 1:
          return n
      else:
          return fibonacci(n-1) + fibonacci(n-2)



  I'll ask what you'd like to do next.
```

無事ローカルでの生成が出来ました！短い例でしたが、体感の生成スピードが上がった気がします。

### Code Llama での生成

無事設定できたわけですが、想像していた通り GPT-4 のように期待通りのアウトプットは中々できません。

```
> 日経225の価格を2018年から2022年までmatplotlibでplotしてください。

  import matplotlib.pyplot as plt
  import pandas as pd

  # Read data from CSV file
  df = pd.read_csv('日経225.csv')

  # Create a line plot
  plt.plot(df['Date'], df['Close'])

  # Set title and labels
  plt.title('日経225の価格')
  plt.xlabel('年')
  plt.ylabel('円')

  # Show the plot
  plt.show()


  File "<stdin>", line 8, in <module>
  File
  "/Users/alexishida/miniforge3/envs/py311/lib/python3.11/site-packages/pandas/io/parsers/readers.py",
  line 948, in read_csv
  return _read(filepath_or_buffer, kwds)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File
  "/Users/alexishida/miniforge3/envs/py311/lib/python3.11/site-packages/pandas/io/parsers/readers.py",
  line 611, in _read
  parser = TextFileReader(filepath_or_buffer, **kwds)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File
  "/Users/alexishida/miniforge3/envs/py311/lib/python3.11/site-packages/pandas/io/parsers/readers.py",
  line 1448, in __init__
  self._engine = self._make_engine(f, self.engine)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File
  "/Users/alexishida/miniforge3/envs/py311/lib/python3.11/site-packages/pandas/io/parsers/readers.py",
  line 1705, in _make_engine
  self.handles = get_handle(
  ^^^^^^^^^^^
  File "/Users/alexishida/miniforge3/envs/py311/lib/python3.11/site-packages/pandas/io/common.py", line
  863, in get_handle
  handle = open(
  ^^^^^


  it seems that the file "日経225.csv" does not exist in the current working directory.
```

## 結論

- OpenInterpreter はローカルの Code Llama を使うことができます。
- ただ、Metal を使うためには、Metal サポートされた Conda で Llama.cpp をインストールする必要があります。
- また、OpenInterpreter のダウンロード時には python 3.10 以上が必要でした。
- 生成物はモデルが Code Llama だとまだまだ微妙。

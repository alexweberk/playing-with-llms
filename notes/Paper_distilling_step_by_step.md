https://blog.research.google/2023/09/distilling-step-by-step-outperforming.html

## 概要

```
[...] distilling step-by-step mechanism enables a 770M parameter T5 model to outperform the few-shot prompted 540B PaLM model using only 80% of examples in a benchmark dataset, which demonstrates a more than 700x model size reduction with much less training data required by standard approaches.
```

Distilling step-by-step を使うと 770M パラメータの T5 モデルが 540B パラメータの PaLM モデルをアウトパフォームすることができる。しかも、ベンチマークデータセットの 80%の例だけを使っている。これは、標準的なアプローチに比べて 700 倍以上のモデルサイズの削減となる。

## 手法の説明

1. 大きいモデルを使ってデータセットを生成する

- CoT を使って(1) input (2) rationale (3) output の３つ構成の例文をプロンプトにいれ、同じパターンのアウトプットを生成させる

2. 小さいモデルのマルチタスクトレーニング

- rationale 生成タスク
- label 予測タスク (４択問題の解答がどれか、等)

## まとめ

Google Research: 770M パラメータの T5 モデルを使って、540B パラメータの PaLM を上回る成果を達成

"Distilling step-by-step" と呼ぶ手法で 540B の PaLM に教師データを作らせ、T5 にマルチタスクトレーニングをしたとのこと。

ステップ:

1. 大きいモデルを使ってデータセットを生成する

- CoT を使って ① インプット、② 結論に至る思考プロセス("rationale")、③ アウトプットの３つ構成の例文をプロンプトにいれ、同じパターンのアウトプットを生成させる

2. 小さいモデルのマルチタスクトレーニング

- rationale 生成タスク
- label 予測タスク(４択問題の解答がどれか、等)

大きいモデルで生成した rationale の中には物事の理解に役立つ情報が結構あって、これが蒸留プロセスの際に役立つという考察。

最近の「モデルサイズ以上に品質の高いデータセットが重要」という潮流に乗った内容でした。

https://blog.research.google/2023/09/distilling-step-by-step-outperforming.html

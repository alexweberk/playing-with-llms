# OpenAI API を使う際の timeout 処理の仕方について

## 問題

OpenAI API を使って大量の処理を行っていた際に、API がタイムアウトしてしまうことがあった。適切な時間でタイムアウトを設定することで、この問題を解決したい。

## 解決策

- retry ライブラリを使って timeout を管理
- `request_timeout` パラメーターを使って timeout を管理

timeout はリクエストする側で独自で運用するべき。

例えば Python の [`retry`](https://pypi.org/project/retry/) ライブラリを使って、

```
@retry(delay=1, backoff=2, max_delay=60)
def func():
    pass
```

等とする。

なぜなら、**API がタイムアウトが起きている状況というのは API 側のサーバーが高負荷になっている状況であり、API サーバー側がリクエスト自体を見れていない可能性すらあるため。**

また、ついでに `request_timeout` パラメーターも入れておくと良い。

```
res = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    request_timeout=60, # 60秒でタイムアウト
    ...
)
```

## 調査

`openai.Completion.create()` を使う際に簡単に `timeout` を設定する方法はないか調査を行った。

[こちらのページ](https://zenn.dev/sion_pn/articles/5373a8db399cce)では、`openai.Completion.create()` に `timeout` という引数が使えるという報告があったが、
残念ながら実際に使ってみると依然タイムアウトが起き、引数が使えていないことがわかった。

調べると[timeout が無視されてしまうバグがあるとの報告](https://github.com/openai/openai-python/issues/322)も報告されていた。[こちら](https://www.bioerrorlog.work/entry/openai-python-timeout)でもその内容が書かれておりました。

他にも `request_timeout` という変数があるとの[コメント](https://github.com/openai/openai-python/issues/322#issuecomment-1476999882)もあった。
ただし、この変数はドキュメントされていないため、今後保守されていくかは不明なため、あまり使うべきじゃないと思われる。

もう少し調べていくと OpenAI のデベロパーコミュニティの[こちらの回答](https://community.openai.com/t/configuring-timeout-for-chatcompletion-python/107226/5?u=alex.ishida)にたどり着きました。曰く、

```
Hi @tdenton8772

Actually, normally managing the timeout for a client-server transitions, like the OpenAPI APIs calls occurs in the client-side code.

This is true of just about every API call (not just OpenAI).

For example when the network is congested, the server might never see the API call, so the timeout will be managed by the client.

Extending this to OpenAI models, when their model is congested, it might not even return an error message (as we have been seeing today), and so it remains the responsibility of the client-side to manage the timeout, which is subtask of overall error and exceptions handling.

HTH

~ 以下GPT-4翻訳 ~
こんにちは @tdenton8772

実際には、クライアントとサーバー間の遷移（OpenAPI APIの呼び出しのような）でタイムアウトを管理するのは通常、クライアント側のコード内で行われます。

これはほぼすべてのAPI呼び出し（OpenAIだけでなく）に当てはまります。

例えば、ネットワークが混雑している場合、サーバーはAPI呼び出しを一切見かけないかもしれませんので、タイムアウトはクライアント側で管理されます。

これをOpenAIのモデルに拡張すると、そのモデルが混雑している場合、エラーメッセージすら返さないかもしれません（今日見てきたように）。したがって、タイムアウトを管理するのは全体のエラーおよび例外処理の一部としてクライアント側の責任となります。

お役に立てれば
```

ということで非常に納得いく回答でした。
納得も行ったので、この回答をもとに `retry` ライブラリを使って timeout を管理することにしようと思った矢先、[こちらの OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb)も見つけました。内容は似た内容だったので本記事では割愛します。

`request_timeout` のパラメーターはドキュメント化されていないため使わないほうが良いと思っていたのですが、[とても重要だと言っているコメント](https://community.openai.com/t/configuring-timeout-for-chatcompletion-python/107226/27?u=alex.ishida)もありました。
ソースコードをたどると Python の一般的な `requests` ライブラリの `requests.Session` に注ぎ込まれているパラメーターのようです。

- [Source](https://github.com/openai/openai-python/blob/7fba4dce4b7e797e708aac7dc1c73a7c66f006d3/openai/api_requestor.py#L82C31-L82C31)
- [Source](https://github.com/openai/openai-python/blob/7fba4dce4b7e797e708aac7dc1c73a7c66f006d3/openai/api_requestor.py#L596C30-L596C30)

一旦私のユースケースでは設定しておいて損はなさそうだったので設定することにしました。
また、同コメントはリクエストのパラレル化についても述べていたのですが今回は割愛しました。

## 結論

OpenAI の API を叩く際には `retry` のライブラリなどを使って timeout を管理しましょう。ついでに `request_timeout` も使っておきましょう。

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

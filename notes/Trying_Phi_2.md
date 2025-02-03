# Microsoft から出た小型モデル Phi-2 を Colab で試す

Microsoft から出た小型モデル Phi-2 を使ってみたいと思います。

- Model: https://huggingface.co/microsoft/phi-2

Satya さんがオープンソースにすると言っていたらしいですが、Huggingface のライセンスはまだリサーチのみ許可するライセンスでした。
[MICROSOFT RESEARCH LICENSE TERMS](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE)

## セットアップ

```python
!pip install accelerate einops -Uqq
```

accelerate が必須だったかわかりませんが、 einops は必要でした。

```python
import torch
print(torch.version)
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
```

    <module 'torch.version' from '/usr/local/lib/python3.10/dist-packages/torch/version.py'>
    2.1.0+cu118
    11.8
    8700
    True

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

```

サンプルコードを試します。

```python
inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

生成を楽にできるようテンプレートとなる関数を作ります。

```python
# プロンプトの準備
def ask(
    prompt: str,
    system_prompt: str = None,
    return_output: bool = False,
) -> str | None:
    if system_prompt:
        prompt = system_prompt + '\n\n' + prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            temperature=0.01,
            repetition_penalty=1.1,
            top_p=0.95,
            do_sample=True,
            streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
        )
    if return_output:
        return outputs
```

## 生成を試す

英語用のモデルなので、英語で試します。

```python
ask("Give me a short answer. What are the first 10 numbers in the fibonacci sequence?")
```

    A: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
    <|endoftext|>

```python
ask("How did the first President of the United States become a President?")
```

    Answer: George Washington was elected by the Electoral College.

    Exercise 4:
    What is the role of the Vice President in the US government?
    Answer: The Vice President serves as the second-in-command and can take over if the President becomes unable to fulfill their duties.

    Exercise 5:
    Why is it important for citizens to participate in elections?
    Answer: Citizens have the power to choose their leaders and shape the future of their country through voting.
    <|endoftext|>

回答は質問に近いですが、求めているのとは違う形で返ってきます。

日本語も試します。

```python
ask("りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？")
```

    Solution:
    To solve this problem, we need to subtract the number of rotten apples from the total number of apples.

    Total number of apples = 5
    Number of rotten apples = 2

    Subtracting the number of rotten apples from the total number of apples gives us:

    5 - 2 = 3

    Therefore, there are 3 good apples left in the basket.

    Follow-up Exercise 1:
    What if there were 8 apples in the basket and 4 of them were rotten? How many good apples would be left?

    Solution to Follow-up Exercise 1:
    Using the same logic as before, we can calculate the number of good apples left by subtracting the number of rotten apples from the total number of apples.

    Total number of apples = 8
    Number of rotten apples = 4

    8 - 4 = 4

    Therefore, there would be 4 good apples left in the basket.

    Follow-up Exercise 2:
    If there were 10 apples in the basket and 6 of them were rotten, how many rotten apples would need to be removed for there to be no rotten apples left?

    Solution to Follow-up Exercise 2:
    To find out how many rotten apples need

全く日本語では答えられないようです。

算数の問題を試します。

```python
ask("A bat and a ball costs $11 total. The bat is $10 more than the ball. How much is the ball")
```

    ?
    <|question_end|>Answer: Let x be the cost of the ball, then the cost of the bat is x + 10.
    The total cost of the bat and ball is $11, so we have:
    x + (x + 10) = 11
    2x + 10 = 11
    2x = 1
    x = 0.5
    Therefore, the cost of the ball is $0.50.
    <|endoftext|>

すんなり正解。

状況把握の問題：

```python
ask("""
### Question
There was a cookie on the table.
Tom entered the room.
The cookie disappeared.
What was likely to have happened?
""")
```

    A. Tom ate the cookie.
    B. The cookie fell off the table and broke.
    C. Someone else took the cookie.
    D. The cookie magically appeared again.

    Solution 0: The correct option is A. Tom ate the cookie.
    Option (a) is the most logical answer because it explains what happened to the cookie in a simple way. Option (b) is unlikely because there is no evidence that the cookie fell or broke. Option (c) is possible, but not very probable, since there is no reason why someone would take the cookie without being noticed. Option (d) is absurd and contradicts the information given in the question.


    <|endoftext|>

プログラミングの問題：

```python
ask("Write me a Python function that parses text to extract multiple email addresses and returns them as a list of strings.")
```

    ```python
    import re

    def extract_emails(text):
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)

    # Example usage:
    text = "Please contact us at info@example.com or support@example.org for assistance."
    print(extract_emails(text)) # Output: ['info@example.com', support@example.org']
    ```
    <|endoftext|>

試しに実行。

```python
import re

def extract_emails(text):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)
```

```python
text = "Please contact us at info.123@example.com or support+gmail@example.org for assistance."
print(extract_emails(text)) # Output: ['info@example.com', support@example.org']
```

    ['info.123@example.com', 'support+gmail@example.org']

ちゃんと動きました。

```python
ask("My EC2 instance cannot git pull from my remote origin. What should I do?")
```

    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._git_pull = None

    @property
    def git_pull(self):
        return self._git_pull

    @git_pull.setter
    def git_pull(self, value):
        if not isinstance(value, bool):
            raise ValueError("GitPull must be a boolean")
        self._git_pull = value

    def _get_git_pull(self):
        try:
            subprocess.check_call(["git", "pull"])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to git pull from {self.remote}")
            raise GitPullError() from e

    def run(self, command, *args, **kwargs):
        """Runs the given command on this EC2 instance."""
        if self.git_pull and not self._git_pull:
            self._get_git_pull()
        return super().run(
    ```

こういうことは答えられないようです。

## おわりに

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回使った Colab:
https://colab.research.google.com/drive/1oS1iMud60uDhQr4txrYuI1BGQIbgmy2s?usp=sharing

## 参考

- 元のモデル: https://huggingface.co/microsoft/phi-2
- ライセンスについて: 商用利用可能な独自ライセンス https://github.com/deepseek-ai/deepseek-LLM/blob/main/LICENSE-MODEL

## 関連

-

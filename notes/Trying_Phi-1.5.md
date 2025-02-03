# Phi 1.5 を試してみた

Microsoft からリリースされた LLM「Phi 1.5」を試してみました。昨今の LLM の中では 1.5B パラメーターと小さいにも関わらず、
各種ベンチマークに関してはかなり高い性能を示しています。今回はこのモデルを使ってみたいと思います。英語のみのモデルでした。

Huggingface: https://huggingface.co/microsoft/phi-1_5
論文: https://arxiv.org/pdf/2309.05463.pdf

## セットアップ

今回は Huggingface にアップされているモデルを使いたいと思います。

モデルカードを見ると、instruction チューニングや RLHF などのトレーニングはしていないと書いてありました。

> We did not fine-tune phi-1.5 either for instruction following or through reinforcement learning from human feedback

生成には下記の 3 種類のフォーマットが推奨されています。[参照](https://huggingface.co/microsoft/phi-1_5#intended-uses)

- QA フォーマット
- Chat フォーマット
- Code フォーマット

## Colab で試してみる

Colab で試してみます。まずは環境をセットアップします。

```
!pip install transformers accelerate einops -q
```

モデルのロード。

```
# Use a pipeline as a high-level helper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device('cuda')
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Model Parameter Count: {model.num_parameters():,.0f}")
```

```
Vocab size: 50257
Model Parameter Count: 1,418,270,720
```

Vocab size 50257 のトークナイザーを使った、1.4B パラメーターのモデルです。

まずは Huggingface のモデルカードにあるサンプルを試してみます。Code フォーマットではインプットとして関数名と関数に行ってほしいことをコメントで渡すフォーマットのようです。

````
inputs = tokenizer('''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
````

````
  \```python
  def print_prime(n):
    """
    Print all primes between 1 and n
    """
    primes = []
    for num in range(2, n+1):
        is_prime = True
        for i in range(2, int(num**0.5)+1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    print(primes)

  print_prime(20)
  \```

  ## Exercises

  1. Write a Python function that takes a list of numbers and returns the sum of all even numbers in the list.

  \```python
  def sum_even(numbers):
    """
    Returns the sum of all even numbers in the list
    """
    return sum(num for num in numbers if
````

ぱっと見あってそうなので実行してみました

```
def print_prime(n):
   """
   Print all primes between 1 and n
   """
   primes = []
   for num in range(2, n+1):
       is_prime = True
       for i in range(2, int(num**0.5)+1):
           if num % i == 0:
               is_prime = False
               break
       if is_prime:
           primes.append(num)
   print(primes)

print_prime(20)
```

```
[2, 3, 5, 7, 11, 13, 17, 19]
```

### QA フォーマット

次に QA フォーマットを試してみます。
QA フォーマットは質問と回答のペアを与えるフォーマットです。質問に対して回答を生成することができます。
インプットは下記のようなパターンです。

```
What is the meaning of life?
Answer:
```

```
What is the meaning of life?
Answer: The meaning of life is a philosophical question that has puzzled humans for centuries.

Exercise 2:
What is the difference between a belief and a fact?
Answer: A belief is something that someone thinks is true, while a fact is something that is proven to be true.

Exercise 3:
What is the difference between a value and a belief?
Answer: A value is something that someone thinks is important, while a belief is something that someone thinks is true.

Exercise 4:
What is the difference between a right and a responsibility?
Answer: A right is something that someone is entitled to, while a responsibility is something that someone is expected to do.

Exercise 5:
What is the difference between a good and a bad thing?
Answer: A good thing is something that is beneficial or helpful, while a bad thing is something that is harmful or detrimental.
<|endoftext|>
```

次の質問

```
How can multi-armed bandit algorithms be integrated into a growth marketing strategy to optimize user engagement?
Answer:
```

```
How can multi-armed bandit algorithms be integrated into a growth marketing strategy to optimize user engagement?
Answer: By using multi-armed bandit algorithms, marketers can analyze user behavior and preferences to create personalized content that resonates with their target audience. This can lead to increased engagement, higher conversion rates, and ultimately, better business outcomes.

Exercise 5:
Exercise: Give an example of how multi-armed bandit algorithms can be used in a real-world scenario.
Answer: Imagine a company that wants to improve its website's conversion rate. By using multi-armed bandit algorithms, they can analyze user behavior, such as the time spent on the website, the pages visited, and the actions taken. This data can then be used to optimize the website's layout, content, and user experience, ultimately leading to a higher conversion rate.

In conclusion, multi-armed bandit algorithms are a powerful tool in the field of machine learning. They allow
```

それなりに意味のある回答が生成されているように見えます。Instruction チューニングされていないので生成は質問への解答以外にも生成されるみたいですね。

### Chat フォーマット

Chat フォーマットは対話を生成するためのフォーマットです。インプットは下記のような `<名前>: <セリフ>` のパターンです。

```
Alice: Hey Bob, how's it going? Still swamped with that machine learning project?

Bob: Ah, you know how it is. Always in the weeds. But I'm making progress. How about you? Still working on that cloud migration?

Alice: Oh, absolutely. It's like trying to change the wheels on a moving car. By the way, have you looked into using Kubernetes? We've started to implement it, and it's a game-changer for container orchestration.

Bob: Kubernetes, huh? Yeah, I've been hearing a lot about it, especially in our DevOps circles. I've been tinkering with it a bit. It's fascinating, but the learning curve is steep. Any tips?

Alice:
```

```
Alice: Definitely! It's like learning to ride a bike. You start with the basics, like understanding the different types of pods and how they interact with each other. Once you grasp
```

自然な感じです。

## まとめ

まだ Instruction tuning 等されていないため、触ってみた感触としては、まずはトレーニングしてなんぼのモデルだと感じました。
パラメーターは 1.5B なので他の LLM と比べてトレーニングコストは低く抑えられそうです。

このサイズにおいてベンチマークですごいパフォーマンスのモデルですが、Twitter での反応の中にはベンチマークテストの内容がリークしてしまっているのではないかと疑う声もありました。
https://twitter.com/suchenzang/status/1701615026648605095

# GPT-4 超えと呼ばれる Xwin-70b を試してみる

今回は AlpacaEval において GPT-4 超えたとされる Xwin-70b を試してみます。

- Huggingface: https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1
- 論文: 未発表
- ライセンス: Llama 2 License

## コードと手順

Colab で試してみます。

まず初めに、Colab 環境で動くよう何かしらの量子化が必要で初め bitsandbytes を使ってみてたのですが、残念ながらパラメーターのデータ量が多すぎて保存領域が足りなくなってしまいました。

そこで、[npaka さんの記事](https://note.com/npaka/n/n83a017e6f621)を参考に、[The Bloke さんが GPTQ 方式で量子化したもの](https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ)をロードすることにしました。

### 必要なライブラリをインストール

```python
!pip install transformers accelerate sentencepiece optimum auto-gptq -Uqq
```

## モデルの用意

GPTQ 方式で量子化された Xwin-70b モデルをロードします。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# , BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

model_id = "TheBloke/Xwin-LM-70B-V0.1-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    # quantization_config=quantization_config,
    device_map='auto',
).eval()
```

```python
# トークナイザーのサイズを確認。
tokenizer.vocab_size
```

> 32000

まずは Huggingface のモデルカードにあるサンプルを走らせてみます。

```python
(
    prompt := "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: Hello, can you help me? "
            "ASSISTANT:"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
samples = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
output = tokenizer.decode(samples[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(output)
```

    Hello! Of course, I'd be happy to help you with any questions or topics you have. Please feel free to ask, and I'll do my best to provide you with useful information and guidance.

まずは英語の解答はとても自然でした。

### 色々と質問してみる

日本語で質問してみたいと思います。

```python
text = """
USER: りんごが5つあります。そこから2つのりんごを取り除きました。残りのりんごの数は何個でしょう？
ASSISTANT:
""".strip()
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    output_ids = model.generate(
        inputs['input_ids'].to(model.device),
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)

```

> <s> USER: りんごが 5 つあります。そこから 2 つのりんごを取り除きました。残りのりんごの数は何個でしょう？
> ASSISTANT: りんごが 5 つあり、そこから 2 つのりんごを取り除いたとされています。それより、残りのりんごの数は 3 個です。</s>

```python
text = """
USER: バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？
ASSISTANT:
""".strip()

inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    output_ids = model.generate(
        inputs['input_ids'].to(model.device),
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)

```

<style>
  pre {
      white-space: pre-wrap;
  }
</style>

    <s> USER: バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。ボールはいくらでしょう？
    ASSISTANT: バットとボールの両方を買うと1100円です。バットはボールよりも1000円高いです。それゆえに、ボールの価格を x として、以下のような方程式を作成できます。

    x + 1000 = 1100

    この方程式を解くと、ボールの価格は100円です。</s>

```python
text = """
USER: 引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。
ASSISTANT:
""".strip()

inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    output_ids = model.generate(
        inputs['input_ids'].to(model.device),
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)

```

<style>
  pre {
      white-space: pre-wrap;
  }
</style>

    <s> USER: 引数kを取り、返り値としてフィボナッチ数列におけるk個目の値を返すPython関数を書いてください。
    ASSISTANT: def fibonacci_number(k):
        if k <= 1:
            return k

        a, b = 0, 1
        for _ in range(k - 2):
            c = a + b
            a, b = b, c

        return b

    # テスト
    k = 5
    print(fibonacci_number(k))
    ```python
    </s>

```python
text = """
USER: 次の内容を日本語に訳してください。"There were 3 apples and 2 oranges. How many fruits were there in total?"
ASSISTANT:
""".strip()

inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    output_ids = model.generate(
        inputs['input_ids'].to(model.device),
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)

```

<style>
  pre {
      white-space: pre-wrap;
  }
</style>

    <s> USER: 次の内容を日本語に訳してください。"There were 3 apples and 2 oranges. How many fruits were there in total?"
    ASSISTANT: それは、「3個のりんごと2個のオレンジがあった。合計でどれくらいの果物があるか？」という内容です。</s>

```python
text = """
USER: 大規模言語モデルについて説明してください。
ASSISTANT:
""".strip()
token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=200,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)
```

<style>
  pre {
      white-space: pre-wrap;
  }
</style>

    USER: 大規模言語モデルについて説明してください。
    ASSISTANT: 大規模言語モデル（Large Language Models、LLM）とは、自然言語処理や人工知能の分野で用いられる、巨大なデータセットを学習したモデルです。これらのモデルは、テキストの生成や理解、情報抽出、翻訳など、多くの応用があります。

    以下に、大規模言語モデルの主な特徴や利用方法を挙げます。

    1. 柔軟性：大規模言語モデルは、自然言語の豊かな表現力を持ち、��

```python
text = """
USER: What are the differences in training methods between Pre-training,  SFT training, Instruction training, PPO training, RHLF training in the context of LLMs?
ASSISTANT:
""".strip()

inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    output_ids = model.generate(
        inputs['input_ids'].to(model.device),
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)

```

<style>
  pre {
      white-space: pre-wrap;
  }
</style>

    <s> USER: What are the differences in training methods between Pre-training,  SFT training, Instruction training, PPO training, RHLF training in the context of LLMs?
    ASSISTANT: In the context of large language models (LLMs), various training methods are employed to enhance their performance and capabilities. Here are the main differences between pre-training, SFT training, instruction training, PPO training, and RHLF training:

    1. Pre-training: This is a stage in which a model is trained on a large dataset to learn general patterns and representations in the data. The most common pre-training task for LLMs is masked language modeling (MLM), where the model learns to predict missing tokens in a text sequence. Other pre-training tasks include next-token prediction, language model fine-tuning (LM-FT), and more. Pre-training lays the foundation for the model's understanding of language and enables it to perform well on downstream tasks with minimal fine-tuning. Examples of pre-trained models include GPT-3, RoBERTa, and BERT.
    2. SFT (Self-Focused Training): SFT is an unsupervised training method that aims to improve the model's ability to understand and generate long-range coherent text. It does this by having the model predict the next token in a sequence but with a twist: the prediction target is chosen from a small window of tokens surrounding the current position, rather than just the previous token. This encourages the model to consider a broader context when making predictions, leading to better performance on tasks like story generation and summarization.
    3. Instruction Training: Instruction training involves fine-tuning a pre-trained model on a dataset containing human-written instructions or demonstrations. The goal is to teach the model to follow specific directions or imitate certain styles, improving its performance on tasks that require understanding and generating text based on given instructions. For example, instruction training can be used to teach a model to answer questions, complete sentences, or generate text in a specific style or tone.
    4. PPO (Proximal Policy Optimization) Training: PPO is a reinforcement learning algorithm used to train agents to perform tasks in environments with delayed rewards. In the context of LLMs, PPO training involves fine-tuning the model to maximize a reward function that measures the quality of the generated text. This can be applied to tasks like dialogue generation, where the model learns to generate responses that are both relevant and engaging. PPO training can also be combined with other techniques, such as imitation learning, to further improve the model's performance.
    5. RHLF (Reinforced Human-like Feedback) Training: RHLF is a training method that combines reinforcement learning with human feedback to guide the model towards generating more human-like text. In RHLF training, a pre-trained model is fine-tuned using a reward function that measures how closely the generated text resembles human-written text. This is achieved by having humans provide feedback in the form of rewards or penalties for each generated token. The model then learns to optimize its output to maximize the total reward, resulting in more human-like text.

    These training methods are not mutually exclusive and can be combined in various ways to improve the performance of LLMs on different tasks. Researchers often experiment with different training strategies and techniques to find the best approach for a specific application or domain.</s>

SFT (Supervised Fine-Tuning) に関しては誤った内容が生成されておりました。

参考までに、gpt-3.5-turbo のアウトプットは以下でした。

```
In the context of Language Model (LM) training, there are several different methods that can be used. Here are the differences between some of the commonly used training methods:

1. Pre-training: Pre-training is the initial phase of training where the LM is trained on a large corpus of unlabeled text data. The objective is to learn the statistical patterns and language representations from this data. Models like GPT (Generative Pre-trained Transformer) use unsupervised learning during pre-training to predict the next word in a sentence or fill in masked words.

2. SFT (Supervised Fine-tuning) training: After pre-training, the LM is fine-tuned on a smaller dataset that is labeled or annotated for a specific task. This fine-tuning process helps the model adapt to the specific task requirements. For example, in the case of text classification, the LM can be fine-tuned on a labeled dataset where each text sample is associated with a specific class label.

3. Instruction training: Instruction training involves training the LM with explicit instructions or demonstrations. The model is provided with examples of desired behavior or specific instructions to follow during training. This method is useful for tasks that require specific guidance, such as question-answering or dialogue systems.

4. PPO (Proximal Policy Optimization) training: PPO is a reinforcement learning algorithm used for training LMs. It involves an agent (the LM) interacting with an environment and receiving rewards or penalties based on its actions. The agent then updates its policy to maximize the expected rewards. PPO training is commonly used for tasks like dialogue generation or reinforcement learning from human feedback.

5. RHLF (Reinforcement Learning from Human Feedback) training: RHLF is a training method that combines supervised fine-tuning with reinforcement learning. Initially, the LM is fine-tuned using supervised learning with human-generated responses as targets. Then, reinforcement learning is applied, where the model interacts with the environment and receives rewards based on its responses. The model is updated to maximize the expected rewards. RHLF training is often used for tasks like chatbot training.

These training methods have different objectives and approaches, and their suitability depends on the specific task and available data. Researchers and practitioners choose the most appropriate method based on the requirements and constraints of their particular LM application.
```

## まとめ

- GPT-4 超えと呼ばれる Xwin-70b を試してみましたが、生成内容の品質は高めではあるものの、主観ベースだと gpt-3.5 にも及んでいない印象でした。
- ただ、プロンプトの仕方があってないなどはあったのかもしれないのでモデルの本領が発揮できたかはわかりません。また、今回試せたのは GPTQ 版ではあるため、それによって品質が低下した可能性も（少しは）あります。
- Colab ＋のプランでも、 70B 規模のモデルともなると GPU のメモリだけではなく Disk 容量の方においても開けるモデルが限られてくる。ぱっと調べた感じだと Colab の Disk 領域を簡単に拡張する方法が見つかりませんでした。

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回の Colab はこちらです：
https://colab.research.google.com/drive/1K4iAZdN6P6GVQ5FlKSp0DUjlalr3yVkY?usp=sharing

## 参考

- https://note.com/npaka/n/n83a017e6f621
- https://huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ
- https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1

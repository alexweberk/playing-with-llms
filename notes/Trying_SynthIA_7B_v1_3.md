# migtissera/SynthIA-7B-v1.3 を Colab で試してみる

今回は Mistral-7B をベースにファインチューニングされた `migtissera/SynthIA-7B-v1.3` を使ってみたいと思います。

- Huggingface: https://huggingface.co/migtissera/SynthIA-7B-v1.3

## 必要なライブラリをインストール

```python
# 必要なライブラリをインストール
!pip install transformers langchain accelerate bitsandbytes -Uqq
```

## コードを実行

必要なライブラリをロードします。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
```

## モデルの用意

今回は `migtissera/SynthIA-7B-v1.3` を BitsandBytes で 4bit に量子化したものを使います。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "migtissera/SynthIA-7B-v1.3"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
).eval()
```

```python
tokenizer.vocab_size

```

    32000

モデルカードのサンプルコードをもとに、少し使いやすいように書き換えてみました。

```python
def generate_text(prompt: str, max_length: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = {
        "do_sample": True,
        "top_p": 1.0,
        "temperature": 0.75,
        "max_length": max_length,
        "top_k": 50,
    }
    outputs = model.generate(
        **inputs,
        **generation_config,
        use_cache=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 質問してみる

英語のモデルなので、まずは英語で質問してみます。

```python
user_input = f"Is the universe expanding? How is that even possible?" # 質問

system_prompt = f"SYSTEM: Elaborate on the topic using a Tree of Thoughts and backtrack when necessary to construct a clear, cohesive Chain of Thought reasoning. Always answer without hesitation."
llm_prompt = f"{system_prompt} \nUSER: {user_input} \nASSISTANT: "

answer = generate_text(llm_prompt)
answer.replace(llm_prompt, '')
```

    " Yes, the universe is indeed expanding, and it's possible because of the concept of dark energy. \n\nEvolved Thought: \nThe expansion of the universe is one of the most profound discoveries in modern cosmology. It was first proposed by astronomer Edwin Hubble in the 1920s, based on his observations of the redshift of light from distant galaxies. This redshift is caused by the stretching of space itself due to the expansion of the universe.\n\nElaborated Thought:\nThe universe is made up of various components, including stars, galaxies, and other celestial bodies. All of these components are held together by the force of gravity. However, the universe is also subject to a mysterious force, known as dark energy, which is causing the expansion of the universe to accelerate. This means that the universe is not only expanding, but it is expanding faster and faster over time.\n\nBacktracking Thought:\nThe concept of dark energy was first proposed in the 1990s by astronomers Michael Turner and Steven Weinberg. Before that, it was believed that the expansion of the universe was slowing down due to the gravitational attraction between all the components of the universe. However, observations of distant supernovae in the 1990s revealed that the expansion of the universe was accelerating, which led to the discovery of dark energy.\n\nAnswer:\nThe expansion of the universe is a fascinating phenomenon that has been discovered through the observations of astronomers and the study of cosmology. It is possible because of the existence of dark energy, which is causing the expansion of the universe to accelerate over time. This discovery has far-reaching implications for our understanding of the universe and the nature of reality."

日本語用のトレーニングはされていませんが、日本語の質問を試してみます。

```python
user_input = f"バイオエンジニアリングの勉強をしたいです。どのような順番で勉強していけばいいですか？"
system_prompt = f"SYSTEM: Elaborate on the topic using a Tree of Thoughts and backtrack when necessary to construct a clear, cohesive Chain of Thought reasoning. Always answer without hesitation. Answer in Japanese."
llm_prompt = f"{system_prompt} \nUSER: {user_input} \nASSISTANT: "

answer = generate_text(llm_prompt)
answer.split("\nASSISTANT: ")[-1]
```

    ' おはようございます。バイオエンジニアリングはいろいろな分野があり、勉強する際にはそれぞれの分野に関する知識を持たなければならないでしょう。それらを学び、それらを組み合わせてバイオエンジニアリングに関する知識を皆得ることができると思います。'

```python
user_input = f"最近朝起きたときからちょっとした体の疲れがあります。どうすればよいでしょう・・・"
system_prompt = f"SYSTEM: Elaborate on the topic using a Tree of Thoughts and backtrack when necessary to construct a clear, cohesive Chain of Thought reasoning. Always answer without hesitation. Answer in Japanese."
llm_prompt = f"{system_prompt} \nUSER: {user_input} \nASSISTANT: "

answer = generate_text(llm_prompt)
answer.split("\nASSISTANT: ")[-1]
```

    ' 最近は朝起きたときから、少しだけ体の疲れがあります。どうすればよいでしょう・・・ \n\nFirst, I will consider the reasons why I feel tired in the mornings. Possible causes could be insufficient sleep, improper diet, or stress. If I have insufficient sleep, it would be helpful to make sure I get enough sleep by going to bed earlier and waking up later. If it is due to poor diet, I should consider adding more nutritious foods to my diet, such as fruits and vegetables. If it is due to stress, I could try to manage my stress levels by practicing relaxation techniques or seeking professional help.\n\nNext, I will consider whether there are any underlying health issues that may be causing the fatigue. If I have been feeling tired for an extended period of time, it may be worth visiting a doctor to rule out any potential health problems.\n\nLastly, I will consider whether my lifestyle habits may be contributing to the fatigue. If my lifestyle is not conducive to a healthy lifestyle, I should try to make changes to my daily routine to incorporate more physical activity and reduce sedentary behavior. \n\nTo summarize, I should make sure I get enough sleep, maintain a healthy diet, manage my stress levels, rule out any underlying health issues, and make lifestyle changes to reduce fatigue.'

日本語で返せと指示しているものの英語で返ってきました。折角なので、翻訳に使えるか試してみましょう。

```python
user_input = f"First, I will consider the reasons why I feel tired in the mornings. Possible causes could be insufficient sleep, improper diet, or stress. If I have insufficient sleep, it would be helpful to make sure I get enough sleep by going to bed earlier and waking up later. If it is due to poor diet, I should consider adding more nutritious foods to my diet, such as fruits and vegetables. If it is due to stress, I could try to manage my stress levels by practicing relaxation techniques or seeking professional help.\n\nNext, I will consider whether there are any underlying health issues that may be causing the fatigue. If I have been feeling tired for an extended period of time, it may be worth visiting a doctor to rule out any potential health problems.\n\nLastly, I will consider whether my lifestyle habits may be contributing to the fatigue. If my lifestyle is not conducive to a healthy lifestyle, I should try to make changes to my daily routine to incorporate more physical activity and reduce sedentary behavior. \n\nTo summarize, I should make sure I get enough sleep, maintain a healthy diet, manage my stress levels, rule out any underlying health issues, and make lifestyle changes to reduce fatigue."
system_prompt = f"SYSTEM: Translate the USER input to Japanese."
llm_prompt = f"{system_prompt} \nUSER: {user_input} \nASSISTANT: "

answer = generate_text(llm_prompt, max_length=1024)
answer.replace(llm_prompt, '')
```

    ' 最初、私は午前に疲れる理由を考えます。原因として、不足睡眠、不正確な食事、ストレスが考えられます。不足睡眠でなければ、私は早く寝ると遅く起きるようにすべきです。不正確な食事が原因であれば、私はフルーツ、野菜等のHealthfulな食材を食べるようにします。ストレスが原因であれば、私はストレスレベルの管理を行う方法または医師の助けを求めるべきです。\n\n次に、私はある場合でも、体内の問題が疲れの原因である可能性を考えます。若干長期間の疲れを感じていた場合、医師を訪問し、健康問題を排除することを考えるべきです。\n\n最後に、私の生活スタイルが疲れの原因となっているかを考えます。若干健康生活スタイルに対応していない場合、私はダイリーのルチャーによって活動を増やし、躊躇生活を改善するようにすべきです。\n\nまとめると、私は睡眠時間を充分に取って、健康な食材を食べ、ストレスレベルを管理するようにする必要があります。私の生命スタイルに問題があれば、それを改善する必要があります。'

意味は伝わりますが、正しい日本語ではないものが返ってきました。

## まとめ

いいモデルですがやはり日本語トレーニングされていないモデルだと日本語は難しいですね。

今回使った Colab: https://colab.research.google.com/drive/1-dHiF_Z2GcbT9SKMYyOc5s6pUm7IJbkH?usp=sharing

以上、お読みいただきありがとうございます。少しでも参考になればと思います。

もし似たようなコンテンツに興味があれば、フォローしていただけると嬉しいです：

- [note](https://note.com/alexweberk/) と
- [Twitter](https://twitter.com/alexweberk)

https://twitter.com/alexweberk

今回の Colab はこちらです：
https://colab.research.google.com/drive/1CJQ80Wm_e443WIf9QKknyU4Ryh-STfBG#scrollTo=gp5p-C4Rv62W

## 参考

- https://huggingface.co/migtissera/SynthIA-7B-v1.3
- https://huggingface.co/mistralai/Mistral-7B-v0.1

## 関連

https://note.com/alexweberk/n/nf8375c5a1fa9
https://note.com/alexweberk/n/nfb5f361ff718
https://note.com/alexweberk/n/n439fc8264668

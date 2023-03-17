from transformers import pipeline

chef = pipeline(
    "text-generation", model="./gpt2-train", tokenizer="rinna/japanese-gpt2-medium"
)
print(chef("U:最近どうですか？")[0]["generated_text"])

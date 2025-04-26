
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

model_path = "./flan-t5-small-finetuned"

# Load tokenizer and model manually from local path
tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

# Create the pipeline with the loaded model and tokenizer
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)



while True:
    prompt=input('>>>>')
    if prompt.lower() =='exit':
        break
    out=pipe(prompt,max_new_tokens=50)
    print("\n Response:", out[0]['generated_text'])
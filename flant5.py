from transformers import T5Tokenizer,T5ForConditionalGeneration
import os

model_name="google/flan-t5-small"
local_dir="./models/flan"

print("downloading....")

tokenizer=T5Tokenizer.from_pretrained(model_name)
model=T5ForConditionalGeneration.from_pretrained(model_name)


os.makedirs(local_dir,exist_ok=True)
print(f"saving model to {local_dir}")
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print("loading model")
tokenizer=T5Tokenizer.from_pretrained(local_dir)
model = T5ForConditionalGeneration.from_pretrained(local_dir)

while True:
        prompt= input()

        inputs=tokenizer(prompt,return_tensors='pt')


        print("generating rspnse")
        outputs = model.generate(**inputs, max_new_tokens=50)
        response=tokenizer.decode(outputs[0],skip_special_tokens=True)


        print('\n Test Output')
        print(response)
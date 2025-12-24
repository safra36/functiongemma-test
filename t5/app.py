from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from PIL import Image
import requests

# Load model
processor = AutoProcessor.from_pretrained("google/t5gemma-2-270m-270m")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-270m-270m")

# With image
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# prompt = "<start_of_image> in this image, there is"

# inputs = processor(text=prompt, images=image, return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=50)
# print(processor.decode(outputs[0]))

# Text-only
text_prompt = "Translate this into persian: Hi What is your name?"
inputs = processor(text=text_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=500)
print(processor.decode(outputs[0]))
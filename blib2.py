import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Fra_Filippo_Lippi_014.jpg/548px-Fra_Filippo_Lippi_014.jpg' 
print("Img url: " + url)
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')   

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# by default `from_pretrained` loads the weights in float32
# we load in float16 instead to save memory
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16) 

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)



def run_prompt(prompt, question=False):
    if question:
        prompt = "Question: " + prompt + " Answer: "
    print("Prompt:" + prompt)
    if prompt == "":
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor(image, text="" + prompt + " A", return_tensors="pt").to(device, torch.float16)


    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("Result: " + generated_text)
    return(generated_text)

run_prompt("")
run_prompt("is this a painting", question=True)
run_prompt("the image is composed of")
run_prompt("the colorscheme is")
run_prompt("the iconography of the painting is")
run_prompt("the painting invokes feelings of")
run_prompt("the art historical context of this painting is")
run_prompt("in the painting is something special at the position of")
run_prompt("what is the most prominent object in the painting")
run_prompt("what object in the painting is missing", question=True)
run_prompt("which architecture fits to the painting", question=True)
run_prompt("how many elements has the painting", question=True)
run_prompt("what is the prominent feeling attached to the painting", question=True)
run_prompt("what in the painting is scary", question=True)
run_prompt("in the future the objects in the painting will")
run_prompt("the most interesting object in the painting is")
run_prompt("in the center of the painting is")
run_prompt("the red object in the painting is")
run_prompt("the most frequent colour is")
run_prompt("you find information about the painting in the book with the title")
run_prompt("three important iconographic features are")
run_prompt("everyone likes about the painting that")
run_prompt("it is strange in the painting that")
run_prompt("at the edges of the painting there are")
run_prompt("in the center of the painting there is")
run_prompt("what kinds of animals are in the painting", question=True)
run_prompt("nobody knows that in the painting you find")
run_prompt("it is lovely that there are so many")

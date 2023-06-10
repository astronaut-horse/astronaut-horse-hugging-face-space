# ----- Deployment Log -----------------------------------------------------------------

# added beta 4305ed7
# added beta 4307f62
# added presidents beta
# added painting concept
# added presidents concept
# added presidents concept #2
# added philip guston concept (retry)
# added Ken Price trainings (retry)
# added Andrei Tarkovsky polaroid training 
# added Andrei Tarkovsky polaroid training (retry)
# added HairBot training
# redeploy with canny edge tab
# try to redeploy
# try to redeploy again

# ----- General Setup -----------------------------------------------------------------

import requests
import os
import gradio as gr
import wget
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from huggingface_hub import HfApi
from transformers import CLIPTextModel, CLIPTokenizer
import html
import datetime

image_count = 0

community_icon_html = ""

loading_icon_html = ""
share_js = ""

api = HfApi()
models_list = api.list_models(author="sd-concepts-library", sort="likes", direction=-1)
models = []

my_token = os.environ['api_key']

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", revision="fp16", torch_dtype=torch.float16, use_auth_token=my_token).to("cuda")


def check_prompt(prompt):
    SPAM_WORDS = ['Д', 'oob', 'reast'] # only necessary to limit spam
    for spam_word in SPAM_WORDS:
        if spam_word in prompt:
            return False
    return True


def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
  
  _old_token = token
  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embeds = loaded_learned_embeds[trained_token]

  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  
  # add the token in tokenizer
  token = token if token is not None else trained_token
  num_added_tokens = tokenizer.add_tokens(token)
  i = 1
  while(num_added_tokens == 0):
    token = f"{token[:-1]}-{i}>"
    num_added_tokens = tokenizer.add_tokens(token)
    i+=1
  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds
  return token



# ----- ControlNet Canny Edges Pipe / Setup -----------------------------------------------------------------

# import gradio as gr
# from PIL import Image
# import numpy as np
# import cv2

# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# from diffusers import UniPCMultistepScheduler
# import torch

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
# controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
# )

# controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
# controlnet_pipe.enable_model_cpu_offload()
# controlnet_pipe.enable_xformers_memory_efficient_attention()





# ----- Load All models / concepts -----------------------------------------------------------------


ahx_model_list = [model for model in models_list if "ahx" in model.modelId]
ahx_dropdown_list = [model for model in models_list if "ahx-model" in model.modelId]


for model in ahx_model_list:
  model_content = {}
  model_id = model.modelId
  model_content["id"] = model_id
  embeds_url = f"https://huggingface.co/{model_id}/resolve/main/learned_embeds.bin"
  os.makedirs(model_id,exist_ok = True)
  if not os.path.exists(f"{model_id}/learned_embeds.bin"):
    try:
      wget.download(embeds_url, out=model_id)
    except:
      continue

  token_identifier = f"https://huggingface.co/{model_id}/raw/main/token_identifier.txt"
  response = requests.get(token_identifier)
  token_name = response.text
  
  concept_type = f"https://huggingface.co/{model_id}/raw/main/type_of_concept.txt"
  response = requests.get(concept_type)
  concept_name = response.text
  model_content["concept_type"] = concept_name
  images = []
  for i in range(4):
    url = f"https://huggingface.co/{model_id}/resolve/main/concept_images/{i}.jpeg"
    image_download = requests.get(url)
    url_code = image_download.status_code
    if(url_code == 200):
      file = open(f"{model_id}/{i}.jpeg", "wb") ## Creates the file for image
      file.write(image_download.content) ## Saves file content
      file.close()
      images.append(f"{model_id}/{i}.jpeg")
  model_content["images"] = images
  #if token cannot be loaded, skip it
  try:
    learned_token = load_learned_embed_in_clip(f"{model_id}/learned_embeds.bin", pipe.text_encoder, pipe.tokenizer, token_name)
    # _learned_token_controlnet = load_learned_embed_in_clip(f"{model_id}/learned_embeds.bin", controlnet_pipe.text_encoder, controlnet_pipe.tokenizer, token_name)
  except: 
    continue
  model_content["token"] = learned_token
  models.append(model_content)
  models.append(model_content)


# -----------------------------------------------------------------------------------------------


model_tags = [model.modelId.split("/")[1] for model in ahx_model_list]
model_tags.sort()
import random 

DROPDOWNS = {}

for model in model_tags:
  if model != "ahx-model-1" and model != "ahx-model-2":
    DROPDOWNS[model] = f" in the style of <{model}>"

# def image_prompt(prompt, dropdown, guidance, steps, seed, height, width):
def image_prompt(prompt, guidance, steps, seed, height, width):
  # prompt = prompt + DROPDOWNS[dropdown]
  square_pixels = height * width
  if square_pixels > 640000:
      height = 640000 // width
  generator = torch.Generator(device="cuda").manual_seed(int(seed))

  height=int((height // 8) * 8)
  width=int((width // 8) * 8)

  # image_count += 1
  curr_time = datetime.datetime.now()

  is_clean = check_prompt(prompt)

  print("----- advanced tab prompt ------------------------------")
  print(f"prompt: {prompt}, size: {width}px x {height}px, guidance: {guidance}, steps: {steps}, seed: {int(seed)}")
  # print(f"image_count: {image_count}, datetime: `{e}`")
  print(f"datetime: `{curr_time}`")
  print(f"is_prompt_clean: {is_clean}")
  print("-------------------------------------------------------")

  if is_clean:
    return (
      pipe(prompt=prompt, guidance_scale=guidance, num_inference_steps=steps, generator=generator, height=height, width=width).images[0], 
      f"prompt: '{prompt}', seed = {int(seed)},\nheight: {height}px, width: {width}px,\nguidance: {guidance}, steps: {steps}"
    )
  else:
    return (
      pipe(prompt="", guidance_scale=0, num_inference_steps=1, generator=generator, height=32, width=32).images[0], 
      f"Prompt violates Hugging Face's Terms of Service"
    )

def default_guidance():
  return 7.5

def default_steps():
  return 30

def default_pixel():
  return 768

def random_seed():
  return random.randint(0, 99999999999999) # <-- this is a random gradio limit, the seed range seems to actually be 0-18446744073709551615



def get_models_text():
  # make markdown text for available models...
  markdown_model_tags = [f"<{model}>" for model in model_tags if model != "ahx-model-1" and model != "ahx-model-2"]
  markdown_model_text = "\n".join(markdown_model_tags)

  # make markdown text for available betas...
  markdown_betas_tags = [f"<{model}>" for model in model_tags if "beta" in model]
  markdown_betas_text = "\n".join(markdown_model_tags)

  return f"## Available Artist Models / Concepts:\n" + markdown_model_text + "\n\n## Available Beta Models / Concepts:\n" + markdown_betas_text



# ----- Advanced Tab -----------------------------------------------------------------

with gr.Blocks(css=".gradio-container {max-width: 650px}") as advanced_tab:
  gr.Markdown('''
      # <span style="display: inline-block; height: 30px; width: 30px; margin-bottom: -3px; border-radius: 7px; background-size: 50px; background-position: center; background-image: url(http://www.astronaut.horse/thumbnail.jpg)"></span> Advanced Prompting

      Freely prompt artist models / concepts with open controls for size, inference steps, seed number etc. Text prompts need to manually include artist concept / model tokens which can be found in the welcome tab and beta tab (ie "an alien in the style of <ahx-model-12>"). You can also mix and match models (ie "a landscape in the style of <ahx-model-14> and <ahx-beta-4307f62>>"). To see example images or for more information see the links below.
      <br><br>
      <a href="http://www.astronaut.horse">http://www.astronaut.horse</a>
      <br>
      <a href="https://discord.gg/ZctfW4SvGw">https://discord.com</a><br>
      <br>
  ''')

  with gr.Row():
    prompt = gr.Textbox(label="image prompt...", elem_id="input-text")
  with gr.Row():
    seed = gr.Slider(0, 99999999999999, label="seed", dtype=int, value=random_seed, interactive=True, step=1)
  with gr.Row():
    with gr.Column():
      guidance = gr.Slider(0, 10, label="guidance", dtype=float, value=default_guidance, step=0.1, interactive=True)
    with gr.Column():
      steps = gr.Slider(1, 100, label="inference steps", dtype=int, value=default_steps, step=1, interactive=True)
  with gr.Row():
    with gr.Column():
      width = gr.Slider(144, 4200, label="width", dtype=int, value=default_pixel, step=8, interactive=True)
    with gr.Column():
      height = gr.Slider(144, 4200, label="height", dtype=int, value=default_pixel, step=8, interactive=True)
  gr.Markdown("<u>heads-up</u>: Height multiplied by width should not exceed about 645,000 or an error may occur. If an error occours refresh your browser tab or errors will continue. If you exceed this range the app will attempt to avoid an error by lowering your input height. We are actively seeking out ways to handle higher resolutions!")
  
  go_button = gr.Button("generate image", elem_id="go-button")
  output = gr.Image(elem_id="output-image")
  output_text = gr.Text(elem_id="output-text")
  go_button.click(fn=image_prompt, inputs=[prompt, guidance, steps, seed, height, width], outputs=[output, output_text])
  gr.Markdown("For a complete list of usable models and beta concepts check out the dropdown selectors in the welcome and beta concepts tabs or the project's main website or our discord.\n\nhttp://www.astronaut.horse/concepts")
    

# -----------------------------------------------------------------------------------------------

model_tags = [model.modelId.split("/")[1] for model in ahx_model_list]
model_tags.sort()
import random 

DROPDOWNS = {}

# set a default for empty entries...
DROPDOWNS[''] = ''

# populate the dropdowns with full appendable style strings...
for model in model_tags:
  if model != "ahx-model-1" and model != "ahx-model-2":
    DROPDOWNS[model] = f" in the style of <{model}>"

# set pipe param defaults...
def default_guidance():
  return 7.5

def default_steps():
  return 30

def default_pixel():
  return 768

def random_seed():
  return random.randint(0, 99999999999999) # <-- this is a random gradio limit, the seed range seems to actually be 0-18446744073709551615


def simple_image_prompt(prompt, dropdown, size_dropdown):
  seed = random_seed()
  guidance = 7.5

  if size_dropdown == 'landscape':
      height = 624
      width = 1024
  elif size_dropdown == 'portrait':
      height = 1024
      width = 624
  elif size_dropdown == 'square':
      height = 768
      width = 768
  else:
      height = 1024
      width = 624
      
  steps = 30

  height=int((height // 8) * 8)
  width=int((width // 8) * 8)

  prompt = prompt + DROPDOWNS[dropdown]
  generator = torch.Generator(device="cuda").manual_seed(int(seed))

  curr_time = datetime.datetime.now()
  is_clean = check_prompt(prompt)
    
  print("----- welcome / beta tab prompt ------------------------------")
  print(f"prompt: {prompt}, size: {width}px x {height}px, guidance: {guidance}, steps: {steps}, seed: {int(seed)}")
  print(f"datetime: `{curr_time}`")
  print(f"is_prompt_clean: {is_clean}")
  print("-------------------------------------------------------")

  if is_clean:
    return (
      pipe(prompt=prompt, guidance_scale=guidance, num_inference_steps=steps, generator=generator, height=height, width=width).images[0], 
      f"prompt: '{prompt}', seed = {int(seed)},\nheight: {height}px, width: {width}px,\nguidance: {guidance}, steps: {steps}"
      )
  else:
    return (
      pipe(prompt="", guidance_scale=0, num_inference_steps=1, generator=generator, height=32, width=32).images[0], 
      f"Prompt violates Hugging Face's Terms of Service"
    )

  
  
# ----- Welcome Tab -----------------------------------------------------------------

rand_model_int = 2

with gr.Blocks(css=".gradio-container {max-width: 650px}") as new_welcome:
  gr.Markdown('''
      # <span style="display: inline-block; height: 30px; width: 30px; margin-bottom: -3px; border-radius: 7px; background-size: 50px; background-position: center; background-image: url(http://www.astronaut.horse/thumbnail.jpg)"></span> Stable Diffusion Artist Collaborations

      Use the dropdown below to select models / concepts trained on images chosen by collaborating visual artists. Prompt concepts with any text. To see example images or for more information on the project see the main project page or the discord community linked below. The images you generate here are not recorded unless you save them, they belong to everyone and no one.
      <br><br>
      <a href="http://www.astronaut.horse">http://www.astronaut.horse</a>
      <br>
      <a href="https://discord.gg/ZctfW4SvGw">https://discord.com</a><br>
  ''')

  with gr.Row():
    dropdown = gr.Dropdown([dropdown for dropdown in list(DROPDOWNS) if 'ahx-model' in dropdown], label="choose style...")
    size_dropdown = gr.Dropdown(['square', 'portrait', 'landscape'], label="choose size...")
  prompt = gr.Textbox(label="image prompt...", elem_id="input-text")

  go_button = gr.Button("generate image", elem_id="go-button")
  output = gr.Image(elem_id="output-image")
  output_text = gr.Text(elem_id="output-text")
  go_button.click(fn=simple_image_prompt, inputs=[prompt, dropdown, size_dropdown], outputs=[output, output_text])

# Old Text --> This tool allows you to run your own text prompts into fine-tuned artist concepts from an ongoing series of Stable Diffusion collaborations with visual artists linked below. Select an artist's fine-tuned concept / model from the dropdown and enter any desired text prompt. You can check out example output images and project details on the project's webpage. Additionally you can play around with more controls in the Advanced Prompting tab. <br> The images you generate here are not recorded unless you choose to share them. Please share any cool images / prompts on the community tab here or our discord server!



# ----- Beta Concepts -----------------------------------------------------------------

with gr.Blocks() as beta:
  gr.Markdown('''
      # <span style="display: inline-block; height: 30px; width: 30px; margin-bottom: -3px; border-radius: 7px; background-size: 50px; background-position: center; background-image: url(http://www.astronaut.horse/thumbnail.jpg)"></span> Beta Models / Concepts

      This tool allows you to test out newly trained beta concepts trained by artists. To add your own beta concept see the link below. This uses free access to Google's GPUs but will require a password / key that you can get from the discord server. After a new concept / model is trained it will be automatically added to this tab when the app is redeployed. 
      <br><br>
      <a href="https://colab.research.google.com/drive/1FhOpcEjHT7EN53Zv9MFLQTytZp11wjqg#scrollTo=hzUluHT-I42O">train your own beta model / concept</a>
      <br>
      <a href="http://www.astronaut.horse">http://www.astronaut.horse</a>
      <br>
      <a href="https://discord.gg/ZctfW4SvGw">https://discord.com</a><br>
      <br>
  ''')

  with gr.Row():
    dropdown = gr.Dropdown([dropdown for dropdown in list(DROPDOWNS) if 'ahx-beta' in dropdown], label="choose style...")
    size_dropdown = gr.Dropdown(['square', 'portrait', 'landscape'], label="choose size...")
  prompt = gr.Textbox(label="image prompt...", elem_id="input-text")

  go_button = gr.Button("generate image", elem_id="go-button")
  output = gr.Image(elem_id="output-image")
  output_text = gr.Text(elem_id="output-text")
  go_button.click(fn=simple_image_prompt, inputs=[prompt, dropdown, size_dropdown], outputs=[output, output_text])





# ----- Canny Edge Tab -----------------------------------------------------------------

from PIL import Image
import gradio as gr
import numpy as np
import cv2

# Define a function to process the uploaded image
def canny_process_image(input_image, input_low_threshold, input_high_threshold, input_invert):
   # Convert the input image to a NumPy array
   np_image = np.array(input_image)
   output_image = input_image  # For example, just return the input image
   numpy_image = np.array(output_image)
   # Return the processed image

   # low_threshold = 100
   # high_threshold = 200
   canny_1 = cv2.Canny(numpy_image, input_low_threshold, input_high_threshold)
   canny_1 = canny_1[:, :, None]
   canny_1 = np.concatenate([canny_1, canny_1, canny_1], axis=2)
   if input_invert:
     canny_1 = 255 - canny_1
   canny_2 = Image.fromarray(canny_1)

   return np.array(canny_2)

# Define the input and output interfaces
canny_input_image = gr.inputs.Image()
canny_input_low_threshold = gr.inputs.Slider(minimum=0, maximum=1000, step=1, label="Lower Threshold:", default=100)
canny_input_high_threshold = gr.inputs.Slider(minimum=0, maximum=1000, step=1, label="Upper Threshold:", default=200)
canny_input_invert = gr.inputs.Checkbox(label="Invert Image")

canny_outputs = gr.outputs.Image(type="numpy")

# Create the Gradio interface
canny_interface = gr.Interface(fn=canny_process_image, inputs=[canny_input_image, canny_input_low_threshold, canny_input_high_threshold, canny_input_invert], outputs=canny_outputs, title='Canny Edge Tracing', allow_flagging='never')




# ----- New ControlNet Canny Gradio Setup with Block -----------------------------------------------------------------


# !pip install -qq diffusers==0.14.0 transformers xformers git+https://github.com/huggingface/accelerate.git
# !pip install -qq opencv-contrib-python
# !pip install -qq controlnet_aux
# !pip install -qq opencv-python
# !pip install -qq gradio
# !pip install -qq Pillow
# !pip install -qq numpy

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from PIL import Image
import gradio as gr
import numpy as np
import torch
import cv2

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
controlnet_pipe.enable_model_cpu_offload()


def controlnet_edges(canny_input_prompt, input_image, input_low_threshold, input_high_threshold, input_invert, canny_input_seed, canny_input_rotate, canny_negative_prompt):
    np_image = np.array(input_image)

    output_image = input_image
    numpy_image = np.array(output_image)

    low_threshold = 80
    high_threshold = 100
    canny_1 = cv2.Canny(numpy_image, input_low_threshold, input_high_threshold)
    canny_1 = canny_1[:, :, None]
    canny_1 = np.concatenate([canny_1, canny_1, canny_1], axis=2)
    if input_invert:
      canny_1 = 255 - canny_1

    canny_2 = Image.fromarray(canny_1)
    canny_1 = Image.fromarray(canny_1)

    if canny_input_rotate and int(canny_input_rotate) > 0:
      canny_rotation = 360 - int(canny_input_rotate)
      canny_2 = canny_2.rotate(canny_rotation, resample=Image.BICUBIC)
      canny_1 = canny_1.rotate(canny_rotation, resample=Image.BICUBIC)

    input_width, input_height = canny_2.size

    limit_size = 768
    # limit_size = 32

    # resize image
    if input_width > input_height:
        new_width = min(input_width, limit_size)
        new_height = int(new_width * input_height / input_width)
    else:
        new_height = min(input_height, limit_size)
        new_width = int(new_height * input_width / input_height)
    canny_2 = canny_2.resize((new_width, new_height))
    canny_1 = canny_1.resize((new_width, new_height))

    # resize original input image
    input_resize = np.array(input_image)
    input_resize = Image.fromarray(input_resize)
    input_resize = input_resize.resize((new_width, new_height))
    # make canny image now, after resize
    canny_resize = np.array(input_resize)
    canny_resize = cv2.Canny(canny_resize, input_low_threshold, input_high_threshold)
    canny_resize = canny_resize[:, :, None]
    canny_resize = np.concatenate([canny_resize, canny_resize, canny_resize], axis=2)
    if input_invert:
      canny_resize = 255 - canny_resize
    canny_resize = Image.fromarray(canny_resize)
    # rotate new resized canny image
    if canny_input_rotate and int(canny_input_rotate) > 0:
      canny_rotation = 360 - int(canny_input_rotate)
      canny_resize = canny_resize.rotate(canny_rotation, resample=Image.BICUBIC, expand=True)

    prompt = canny_input_prompt
    generator = torch.Generator(device="cpu").manual_seed(canny_input_seed)

    output_image = controlnet_pipe(
        prompt,
        canny_resize,
        negative_prompt=canny_negative_prompt,
        generator=generator,
        num_inference_steps=20,
    )

    return [canny_resize, output_image[0][0]]
    # return output_image[0][0]

import random 
def random_seed():
  return random.randint(0, 99999999999999)


with gr.Blocks() as canny_blocks_interface:
  gr.Markdown('''
      # <span style="display: inline-block; height: 30px; width: 30px; margin-bottom: -3px; border-radius: 7px; background-size: 50px; background-position: center; background-image: url(http://www.astronaut.horse/thumbnail.jpg)"></span> ControlNet + Canny Edge-Tracing
      This tool allows you to apply a Stable Diffusion text prompt to an existing image composition using an edge-tracing tool called Canny Edge Detector. Note that you cannot currently apply trained artist concepts from the other tabs in this application to this process currently as they were trained using a more recent version of Stable Diffusion. 
      <br><br>
      <a href="https://wikipedia.org/wiki/Canny_edge_detector">https://wikipedia.org/wiki/canny_edge_detector</a>
      <br>
      <a href="http://www.astronaut.horse">http://www.astronaut.horse</a>
      <br>
      <a href="https://discord.gg/ZctfW4SvGw">https://discord.com</a><br>
      <br>
  ''')
  with gr.Row():
    with gr.Column():
      canny_input_prompt = gr.inputs.Textbox(label="enter your text prompt here")
      with gr.Accordion(label='negative prompt (optional)', open=False):
        canny_negative_prompt = gr.inputs.Textbox()
      canny_input_low_threshold = gr.inputs.Slider(minimum=0, maximum=1000, step=1, label="Lower Threshold:", default=100)
      canny_input_high_threshold = gr.inputs.Slider(minimum=0, maximum=1000, step=1, label="Upper Threshold:", default=120)
      canny_input_seed = gr.Slider(0, 99999999999999, label="seed", dtype=int, value=random_seed, interactive=True, step=1)
      canny_input_invert = gr.inputs.Checkbox(label="invert edge tracing image")
      canny_input_rotate = gr.Dropdown([0, 90, 180, 270], label="rotate image (for smartphones)")
    with gr.Column():
      canny_input_image = gr.inputs.Image(label="input image")
      go_button = gr.Button('generate image')
  # with gr.Row():
      with gr.Accordion(label='traced edge image', open=False):
        canny_output_1 = gr.outputs.Image(type="pil", label="traced edges")
  with gr.Row():
    canny_output_2 = gr.outputs.Image(type="pil", label="final image")
  go_button.click(fn=controlnet_edges, inputs=[canny_input_prompt, canny_input_image, canny_input_low_threshold, canny_input_high_threshold, canny_input_invert, canny_input_seed, canny_input_rotate, canny_negative_prompt], outputs=[canny_output_1, canny_output_2])


# canny_blocks_interface.launch(debug=False)




# ----- ControlNet with Depth Map -----------------------------------------------------------------







# ----- Launch Tabs -----------------------------------------------------------------

tabbed_interface = gr.TabbedInterface([new_welcome, advanced_tab, beta, canny_blocks_interface], ["Artbots", "Advanced", "Beta", "ControlNet"])
# tabbed_interface = gr.TabbedInterface([new_welcome, advanced_tab, beta], ["Artbots", "Advanced", "Beta"])
tabbed_interface.launch()

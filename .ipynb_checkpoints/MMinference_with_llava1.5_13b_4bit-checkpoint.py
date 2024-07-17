from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.schema import ImageDocument
from llama_index.core import Document
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import torch
import os
import openai
import fitz
import argparse
from PIL import Image
from udopEmbedding import udopUnimodelEmbedding
# Llava
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer

## Llava model
model_path = "4bit/llava-v1.5-13b-3GB"
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

def llava_gen(image_file, prompt):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, 
                                  max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.split('.')[:-1]
    return image, output


## original eval code
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_pdf(pdf_file="./sample_pdf/llama2.pdf"):
    # Split the base name and extension
    output_directory_path, _ = os.path.splitext(pdf_file)

    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    # Open the PDF file
    pdf_document = fitz.open(pdf_file)

    # Iterate through each page and convert to an image
    for page_number in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_number]

        # Convert the page to an image
        pix = page.get_pixmap()

        # Create a Pillow Image object from the pixmap
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        abs_output_path=os.path.abspath(output_directory_path)
        # Save the image
        image.save(f"{abs_output_path}/page_{page_number + 1}.png")

    # Close the PDF file
    pdf_document.close()

    image_paths = []
    for img_path in os.listdir(output_directory_path):
        image_paths.append(str(os.path.join(output_directory_path, img_path)))
    
    return output_directory_path

def dir_reader(directory):
    # List to store the absolute paths of files
    files_list = []

    # Walk through all directories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct the absolute path and add to the list
            files_list.append(os.path.abspath(os.path.join(root, file)))

    return [Document(text=file_name, extra_info= {}) for file_name in files_list]

def build_ragSystem(img_paths):
    # Read the images
    documents_images = dir_reader(img_paths)
    embed_model = udopUnimodelEmbedding(embed_batch_size=1)
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    # if running for the first time, will download model weights first!
    index = VectorStoreIndex.from_documents(documents_images)
    retriever_engine = index.as_retriever(image_similarity_top_k=1)
    return retriever_engine

def qa_system(pdf_path,user_query):
    img_dirs=parse_pdf(pdf_path)
    #openai_mm_llm = OpenAIMultiModal(model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500)
    retriever_engine = build_ragSystem(img_dirs)

    # retrieve for the query using text to image retrieval
    retrieval_results = retriever_engine.retrieve(user_query)

    image_documents = [ImageDocument(image_path=image_path.text) for image_path in retrieval_results]
    print("retrieval_results :",retrieval_results)
    print("type of retrieval_results : ",type(retrieval_results))
    print("image_documents :",image_documents)
    print("type of image_documents : ",type(image_documents))
    # response = openai_mm_llm.complete(
    #     prompt=user_query,
    #     image_documents=image_documents,
    # )
    # #FIXME: image_path is not defined and not work
    # image_path = image_documents[0].image_path
    # image, response = llava_gen(image_path, user_query)
    # print(f"Question: {user_query}")
    # print(f'retrieved page: {retrieval_results}')
    # print(f"response: {response}")

if __name__ == "__main__":
    #parse system arguments
    args = argparse.ArgumentParser()
    args.add_argument('--pdf_path', type=str, default='/home/yejinhand/project/UDOP_RAG/llama2-3.pdf', help='path to pdf file')
    args.add_argument('--query', type=str, default='what is llama2', help='query to be asked')
    args = args.parse_args()

    #call the qa_system function
    qa_system(args.pdf_path, args.query)

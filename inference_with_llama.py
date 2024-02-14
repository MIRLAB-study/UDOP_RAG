import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import io
from PIL import Image, ImageDraw
import numpy as np
import csv
import pandas as pd

from torchvision import transforms

from torch import cuda, bfloat16
from transformers import StoppingCriteria, StoppingCriteriaList, LlamaTokenizer, LlamaForCausalLM
from langchain.llms import HuggingFacePipeline
from peft import PeftModel, PeftConfig
import transformers
import torch

from transformers import AutoModelForObjectDetection
import torch
import openai
import os
import fitz
import qdrant_client
from PIL import Image
import llama_index

from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument

from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

device = "cuda" if torch.cuda.is_available() else "cpu"
OPENAI_API_TOKEN = "your-toekn"
openai.api_key = OPENAI_API_TOKEN
model_id = 'meta-llama/Llama-2-13b-chat-hf'
hf_auth = "<your hf token here>"

pdf_file = "you_pdf_file"

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break

def createHFpipline(model_id, hf_auth, device='cpu',cache_dir=None, peft=False, load_in_8bit=False):
    
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    
    if peft:
        config =  PeftConfig.from_pretrained(model_id,use_auth_token=hf_auth,cache_dir=cache_dir)
        model = transformers.LlamaForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map='auto',
            cache_dir=cache_dir,
            use_auth_token=hf_auth
        )
        model = PeftModel.from_pretrained(
            model,
            model_id=model_id,
        )
    
    else:
        # begin initializing HF items, you need an access token
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth,
            cache_dir=cache_dir
        )
    
    

    # enable evaluation mode to allow model inference
    model.eval()
    print(f"Model loaded on {device}")
    
    if peft:
        tokenizer = LlamaTokenizer.from_pretrained(model_id,cache_dir=cache_dir,use_auth_token=hf_auth)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
    else:
        # begin initializing tokenizer. Again, you need an access token
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_auth,
            cache_dir=cache_dir
        )
    
    
    #generate stopping criteria 
    #specify when the model should stop generating text
    stop_list = ['\nHuman:', '\n```\n', '\n\n\n','\n\n']

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]    
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    # define custom stopping criteria object
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    
    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.3  # without this output begins repeating
    )

    return HuggingFacePipeline(pipeline=generate_text)

def main():
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
  
      # Save the image
      image.save(f"./{output_directory_path}/page_{page_number + 1}.png")
  
  # Close the PDF file
  pdf_document.close()
  
  image_paths = []
  for img_path in os.listdir("./llama2"):
      image_paths.append(str(os.path.join("./llama2", img_path)))
  
  
  plot_images(image_paths[9:12])
  
  llama_llm = createHFpipline(model_id='/home/jerry0110/medAlpaca/llama2_vicuna_med_1017', hf_auth=hf_auth, device=device, cache_dir='/scratch/jerry0110/hf_cache',peft=True, load_in_8bit=True)
  
  # Read the images
  documents_images = SimpleDirectoryReader("./llama2/").load_data()
  
  # Create a local Qdrant vector store
  client = qdrant_client.QdrantClient(path="qdrant_index")
  
  text_store = QdrantVectorStore(
      client=client, collection_name="text_collection"
  )
  image_store = QdrantVectorStore(
      client=client, collection_name="image_collection"
  )
  storage_context = StorageContext.from_defaults(
      vector_store=text_store, image_store=image_store
  )
  
  # Create the MultiModal index
  index = MultiModalVectorStoreIndex.from_documents(
      documents_images,
      storage_context=storage_context,
  )
  
  retriever_engine = index.as_retriever(image_similarity_top_k=2)
  
  from llama_index.core.indices.multi_modal.retriever import (
      MultiModalVectorIndexRetriever,
  )
  
  query = "Compare llama2 with llama1?"
  assert isinstance(retriever_engine, MultiModalVectorIndexRetriever)
  # retrieve for the query using text to image retrieval
  retrieval_results = retriever_engine.text_to_image_retrieve(query)
  
  retrieved_images = []
  for res_node in retrieval_results:
      if isinstance(res_node.node, ImageNode):
          retrieved_images.append(res_node.node.metadata["file_path"])
      else:
          display_source_node(res_node, source_length=200)
  
  plot_images(retrieved_images)
  image_documents = [
    ImageDocument(image_path=image_path) for image_path in retrieved_images
  ]

  response = openai_mm_llm.complete(
    prompt="Compare llama2 with llama1?",
    image_documents=image_documents,
  )

  print(response)

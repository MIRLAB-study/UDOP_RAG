import sys
from typing import Any, List

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.schema import ImageDocument
from llama_index.core import Document
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from transformers import AutoProcessor, AutoModel
import torch
import os
import openai
import fitz
import argparse
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
OPENAI_API_TOKEN = "your api key here"
openai.api_key = OPENAI_API_TOKEN

class udopUnimodelEmbedding(BaseEmbedding):
    _model  = PrivateAttr()
    _processor  = PrivateAttr()
    _processor_no_ocr = PrivateAttr()
    def __init__(
        self,
        udop_model_name: str = "microsoft/udop-large",
        **kwargs: Any,
    ) -> None:
        self._model = AutoModel.from_pretrained(udop_model_name)
        self._processor = AutoProcessor.from_pretrained(udop_model_name, apply_ocr=True)
        self._processor_no_ocr = AutoProcessor.from_pretrained(udop_model_name, apply_ocr=False)
        super().__init__(**kwargs)
        
    def adjust_list_length(self, lst: List[float], length: int) -> List[float]:
        if len(lst) < length:
            # Pad the list with None
            lst += [0] * (length - len(lst))
        elif len(lst) > length:
            # Slice the list to the specified length
            lst = lst[:length]
        return lst

    @classmethod
    def class_name(cls) -> str:
        return "udop_unimodel"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        text=query.split()
        bbox=[[200, 300, 250, 450] for i in text]
        image = Image.new('RGB', (800, 600), 'white')
        inputs = self._processor_no_ocr(image, text, boxes=bbox, return_tensors="pt",padding=True)
        last_hidden_state= self._model.encoder(**inputs).last_hidden_state
        return self.adjust_list_length(last_hidden_state.mean(dim=1).squeeze().tolist(),800000)

    def _get_text_embedding(self, text: str) -> List[float]:
        image=Image.open(text).convert("RGB")
        inputs = self._processor(image, return_tensors="pt",padding=True)
        last_hidden_state= self._model.encoder(**inputs).last_hidden_state
        return self.adjust_list_length(last_hidden_state.mean(dim=1).squeeze().tolist(),800000)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        images=[Image.open(text).convert("RGB") for text in texts if text.endswith(".png")]
        inputs = self._processor(images, return_tensors="pt",padding=True)
        last_hidden_state= self._model.encoder(**inputs).last_hidden_state
        output = last_hidden_state.reshape(last_hidden_state.shape[0],-1).tolist()
        return [ self.adjust_list_length(out,800000) for out in output]

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
    openai_mm_llm = OpenAIMultiModal( model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500)
    retriever_engine = build_ragSystem(img_dirs)

    # retrieve for the query using text to image retrieval
    retrieval_results = retriever_engine.retrieve(user_query)

    image_documents = [ImageDocument(image_path=image_path.text) for image_path in retrieval_results]
    response = openai_mm_llm.complete(
        prompt=user_query,
        image_documents=image_documents,
    )

    print(f"Question: {user_query}")
    print(f'retrieved page: {retrieval_results}')
    print(f"response: {response}")

if __name__ == "__main__":
    #parse system arguments
    args = argparse.ArgumentParser()
    args.add_argument('--pdf_path', type=str, default='./llama2-3.pdf', help='path to pdf file')
    args.add_argument('--query', type=str, default='what is llama2', help='query to be asked')
    args = args.parse_args()

    #call the qa_system function
    qa_system(args.pdf_path, args.query)

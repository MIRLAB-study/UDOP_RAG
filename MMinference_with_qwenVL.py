from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import Document
import torch
import os
import fitz
import argparse
from PIL import Image
from udopEmbedding import udopUnimodelEmbedding

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
    torch.manual_seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    
    img_dirs=parse_pdf(pdf_path)
    # openai_mm_llm = OpenAIMultiModal(model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500)
    retriever_engine = build_ragSystem(img_dirs)

    # retrieve for the query using text to image retrieval
    retrieval_results = retriever_engine.retrieve(user_query)

    query = tokenizer.from_list_format([
    {'image': retrieval_results[0].text},
    {'text': user_query},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    
    print(f"Question: {user_query}")
    print(f'retrieved page: {retrieval_results[0].text}')
    print(f"response: {response}")

if __name__ == "__main__":
    #parse system arguments
    args = argparse.ArgumentParser()
    args.add_argument('--pdf_path', type=str, default='./llama2-3.pdf', help='path to pdf file')
    args.add_argument('--query', type=str, default='what is llama2', help='query to be asked')
    args = args.parse_args()

    #call the qa_system function
    qa_system(args.pdf_path, args.query)

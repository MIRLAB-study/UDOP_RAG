from flask import Flask, request, render_template
import time
# for model

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

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

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
    Settings.chunk_size = 1024

    # if running for the first time, will download model weights first!
    index = VectorStoreIndex.from_documents(documents_images)
    retriever_engine = index.as_retriever(image_similarity_top_k=1)
    return retriever_engine


def qa_system(tokenizer, model, pdf_path, user_query):

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
    return response, retrieval_results[0].text


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        # Get the text from the form
        user_text = request.form['user_text']
        pdf_path='/home/yejinhand/project/UDOP_RAG/merged_answer_only.pdf'
        #user_text : user query
        #
        #이제 모델 inference 돌리면 됨! 지금은 time.sleep(5)가 인퍼런스 대신 써져있는데, 얘를 바꿔주면 됨
        #document에 해당하는 output은 output1 변수로 할당하고, answer는 output2 변수로 할당하면 됨
        try:
            doc, ans = qa_system(tokenizer, model, pdf_path , user_text)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return render_template('index.html', output1="Error: Out of memory. Please try a smaller input.", output2="")
      
        # Render the results on the page
        return render_template('index.html', output1=doc, output2= ans)
    
    return render_template('index.html', output1=None, output2=None)

if __name__ == '__main__':

    app.run(debug=True,port=8083)
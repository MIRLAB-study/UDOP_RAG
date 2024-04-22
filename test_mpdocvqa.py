from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import Document
import torch
import os
import json
import argparse
from udopEmbedding import udopUnimodelEmbedding
from metric import anls
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_ragSystem(img_path, img_list):
    # Read the images from the directory
    documents_images = [Document(text=os.path.abspath(os.path.join(img_path, file_name+".jpg")), extra_info= {}) for file_name in img_list]

    embed_model = udopUnimodelEmbedding(embed_batch_size=4)
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024

    # if running for the first time, will download model weights first!
    index = VectorStoreIndex.from_documents(documents_images)
    retriever_engine = index.as_retriever(image_similarity_top_k=3)
    return retriever_engine

def test_mpdocvqa(img_path,json_path):
    torch.manual_seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    
    #read json and get the query
    with open(json_path) as f:
        json_data = json.load(f)

    for idx,wdata in tqdm(enumerate(json_data['data'])):
        user_query = wdata['question']
        img_list=wdata['page_ids']
        relevant_pages = img_list[wdata['answer_page_idx']]
        answer = wdata['answers']
        
        img_dirs=img_path
        retriever_engine = build_ragSystem(img_dirs, img_list)

        system_prompt=f""
        # retrieve for the query using text to image retrieval
        retrieval_results = retriever_engine.retrieve(system_prompt+user_query)
        answers = []
        for retrieval_result in retrieval_results:
            query = tokenizer.from_list_format([
            {'image': retrieval_result.text},
            {'text': user_query},
            ])
            response, history = model.chat(tokenizer, query=query, history=None)
            retrieved_page=retrieval_result.text.split("/")[-1].split(".")[0]
            answers.append({"retrieved_page":retrieved_page,"response":response, "anls":anls(answer, response), "retreive_score": 1 if retrieved_page==relevant_pages else 0})
            # print(f"\nQuestion: {user_query}")
            # print(f'retrieved page: {retrieved_page}')
            # print(f"response: {response}")
            # print(f"answer: {answer}")
            # print(f"ANLS: {anls(answer, response)}")
            # print(f"Relevant page: {relevant_pages}\n\n")
        json_data['data'][idx]['model_response'] = answers
    
    with open(json_path.replace('.json', '_with_response.json' ), 'w') as f:
        json.dump(json_data, f)

if __name__ == "__main__":
    #parse system arguments
    args = argparse.ArgumentParser()
    args.add_argument('--img_path', type=str, help='path to pdf file')
    args.add_argument('--json_path', type=str, help='query to be asked')
    args = args.parse_args()

    #call the qa_system function
    test_mpdocvqa(args.img_path, args.json_path)
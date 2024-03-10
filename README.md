# UDOP_RAG

Use env.yaml for installation
```
conda env create --file env.yaml
conda activate llama_index
pip install git+https://github.com/openai/CLIP.git
```


You also need a copy of udop repository and UDOP model file
```
git clone https://github.com/microsoft/i-Code.git
```
```
cd ./i-Code/i-Code-Doc
wget https://huggingface.co/ZinengTang/Udop/resolve/main/udop-unimodel-large-224.zip
unzip udop-unimodel-large-224.zip
```

Inference_with_qdrant is simpple implementation of multimodal embedding with Qdrant library. We recommand testing your environment with this before running inference_with_udop.py
To run inference, you must pass directory of your pdf and question as system argument. For example, 
```
python inference_with_qdrant.py ./sample_pdf/llama2sss.pdf "Compare llama2 with llama1"
```

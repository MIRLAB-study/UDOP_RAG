# UDOP_RAG

Use env.yaml for installation
```
conda env create --file env.yaml
conda activate udop_rag
conda install -c conda-forge tesseract
```
To run inference, you must pass directory of your pdf and question as system argument. For example, 
```
python inference_with_qdrant.py --pdf_path ./sample_pdf/llama2.pdf --query "Compare llama2 with llama1"
```
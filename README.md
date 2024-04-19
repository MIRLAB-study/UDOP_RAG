# UDOP_RAG

Use env.yaml for installation
```
conda env create --file env.yaml
conda activate udop_rag
conda install -c conda-forge tesseract
```

To run inference with openai mllm model, you must pass directory of your pdf and question as system argument. Note that you need to set the OPENAI_API_TOKEN in your environment variables
```
python MMinference_with_openAI.py --pdf_path ./sample_pdf/llama2.pdf --query "Compare llama2 with llama1"
```

To run inference with local vlm model, you must pass directory of your pdf and question as system argument.  Current version of the project only supports Qwen-vl model.
```
python MMinference_with_qwenVL.py --pdf_path ./sample_pdf/llama2.pdf --query "Compare llama2 with llama1"
```
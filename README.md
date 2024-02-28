# UDOP_RAG

Use env.yaml for installation
```
conda env create --file environment.yaml
```


You also need a copy of udop repository and UDOP model file
```
git clone https://github.com/microsoft/i-Code.git
```
```
wget https://huggingface.co/ZinengTang/Udop/resolve/main/udop-unimodel-large-224.zip
```

Inference_with_qdrant is simpple implementation of multimodal embedding with Qdrant library. We recommand testing your environment with this before running inference_with_udop.py

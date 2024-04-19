import sys
from PIL import Image
from typing import Any, List
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from transformers import AutoProcessor, AutoModel

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
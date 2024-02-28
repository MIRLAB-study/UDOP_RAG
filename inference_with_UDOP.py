import sys
import inference_with_qdrant
form inference_with_qdrant import parse_pdf
from typing import Any, List

sys.path.append('./i-Code/i-Code-Doc/')
from core.models import UdopUnimodelForConditionalGeneration, UdopConfig,  UdopTokenizer
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings

device = "cuda" if torch.cuda.is_available() else "cpu"
OPENAI_API_TOKEN = "your API key"
openai.api_key = OPENAI_API_TOKEN

class udopUnimodelEmbedding(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()
    def __init__(
        self,
        udop_model_path: str = "./i-Code/i-Code-Doc/udop-unimodel-large-224",
        instruction: str = "Represent a document for semantic search:",
        **kwargs: Any,
    ) -> None:
        config = UdopConfig.from_pretrained(udop_model_path)
        tokenizer = UdopTokenizer.from_pretrained(udop_model_path)
        self._model = UdopUnimodelForConditionalGeneration.from_pretrained(udop_model_path)
        self._instruction = instruction
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "udop_unimodel"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings

def build_ragSystem(img_paths):
    
    openai_mm_llm = OpenAIMultiModal( model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500)

    # Read the images
    documents_images = SimpleDirectoryReader(img_paths).load_data()

    embed_model = udopUnimodelEmbedding(embed_batch_size=2)

    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    # if running for the first time, will download model weights first!
    index = VectorStoreIndex.from_documents(documents)
    retriever_engine = index.as_retriever(image_similarity_top_k=2)
    
    return retriever_engine

def qa_system(pdf_path,user_query="Compare llama2 with llama1?"):
    img_dirs=parse_pdf(pdf_path)
    retriever_engine = build_ragSystem(img_dirs)
    assert isinstance(retriever_engine, MultiModalVectorIndexRetriever)
    # retrieve for the query using text to image retrieval
    retrieval_results = retriever_engine.text_to_image_retrieve(user_query)

    retrieved_images = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_images.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)

    image_documents = [ImageDocument(image_path=image_path) for image_path in retrieved_images]
    response = openai_mm_llm.complete(
        prompt=user_query,
        image_documents=image_documents,
    )
    print(f"Question: {user_query}")
    print(f"response: {response}")

if __name__ == "__main__":
    #parse system arguments
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    
    #call the qa_system function
    qa_system(pdf_path, query)

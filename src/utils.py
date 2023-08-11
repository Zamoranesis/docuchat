# %%
import os
import glob
from typing import List, Any
from langchain.document_loaders import (
                                        CSVLoader,
                                        EverNoteLoader,
                                        PyMuPDFLoader,
                                        TextLoader,
                                        UnstructuredEmailLoader,
                                        UnstructuredEPubLoader,
                                        UnstructuredHTMLLoader,
                                        UnstructuredMarkdownLoader,
                                        UnstructuredODTLoader,
                                        UnstructuredPowerPointLoader,
                                        UnstructuredWordDocumentLoader
                                        )
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import gradio as gr
import torch
import json

# Config
with open("../config/vectordb_config.json", "r") as f:
    vectordb_config = json.load(f)

with open("../config/model_config.json", "r") as f:
    model_config = json.load(f)


# Device
# vectordb_config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Map file extensions to document loaders and their arguments
loader_map = {
                # Documents
                ".csv": (CSVLoader, {}),
                ".doc": (UnstructuredWordDocumentLoader, {}),
                ".docx": (UnstructuredWordDocumentLoader, {}),
                ".html": (UnstructuredHTMLLoader, {}),
                ".md": (UnstructuredMarkdownLoader, {}),
                ".pdf": (PyMuPDFLoader, {}),
                ".ppt": (UnstructuredPowerPointLoader, {}),
                ".pptx": (UnstructuredPowerPointLoader, {}),
                ".txt": (TextLoader, {"encoding": "utf8"}),
                # Images
                ".jpg": (UnstructuredImageLoader, {}),
                ".jpeg": (UnstructuredImageLoader, {}),
                ".png": (UnstructuredImageLoader, {})
                # Add more mappings for other file extensions and loaders as needed
}

# %%

class DocumentChatApp:

    def __init__(self) -> None:

        self.chain = None
        self.chat_history: list = []
        self.N: int = 0
        self.count: int = 0
        self.loader_map = loader_map
        self.vectordb_config = vectordb_config
        self.model_config = model_config

    def __call__(self, file: str) -> Any:
        if self.count==0:
           self.chain = self.build_chain(file)
           self.count+=1

        return self.chain

    def load_file(self, file: str) -> Any:
        extension = "." + file.name.rsplit(".", 1)[-1]
        if extension in self.loader_map:
            loader_class, loader_args = self.loader_map[extension]
            loader = loader_class(file.name, **loader_args)

            return loader
        raise gr.Error(message = f"Unsupported file extension '{ext}'")
 
    def process_file(self,file: str):
        file_name = file.name
        loader = self.load_file(file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.vectordb_config["chunk_size"],
                                                       chunk_overlap=self.vectordb_config["chunk_overlap"]
                                                    )
        texts = text_splitter.split_documents(documents)

        return texts, file_name
    
    def build_chain(self, file: str):
        texts, file_name = self.process_file(file)
        #Load embeddings model
        embeddings = HuggingFaceEmbeddings(model_name=self.vectordb_config["emb_model_name"],
                                           model_kwargs={'device': self.vectordb_config["device"]}
                                    )
        # Vector database
        vectordb = FAISS.from_documents(texts,
                                        embeddings
                                        )
        retriever = vectordb.as_retriever(search_kwargs={'k': self.vectordb_config["k"]})
        # Model
        llm = LlamaCpp(**self.model_config)
        chain = ConversationalRetrievalChain.from_llm(llm, 
                                                      retriever=retriever,
                                                      return_source_documents=True
                                                     )

        return chain

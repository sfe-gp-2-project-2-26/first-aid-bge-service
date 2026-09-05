import uvicorn
import tempfile
import os
import torch
import gc
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel

# Docling Imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading BGE-M3 on GPU...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
logger.info("BGE-M3 Loaded Successfully!")

app = FastAPI(title="Docling & BGE-M3 Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmbedRequest(BaseModel):
    texts: List[str]

@app.get("/health")
def health():
    return {"status": "healthy", "ready": True}

@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        if not req.texts:
            return {"dense": [], "sparse": [], "dense_size": 1024}

        # Memory cleanup before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        out = model.encode(
            req.texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        dense_vecs = [v.tolist() for v in out['dense_vecs']]
        sparse_vecs = []
        for lex in out['lexical_weights']:
            sparse_vecs.append({
                "indices": [int(k) for k in lex.keys()],
                "values": [float(v) for v in lex.values()]
            })

        # Memory cleanup after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "dense": dense_vecs,
            "sparse": sparse_vecs,
            "dense_size": 1024
        }
    except Exception as e:
        logger.error(f"Error during /embed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

CHUNK_MAX_TOKENS = 512

# Initialize Docling Converter
pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
docling_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

# Initialize Chunker
tokenizer = HuggingFaceTokenizer.from_pretrained("BAAI/bge-m3")
chunker = HybridChunker(tokenizer=tokenizer, max_tokens=CHUNK_MAX_TOKENS)

@app.post("/chunk_pdf")
@app.post("/parse_pdf")
async def chunk_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        # Memory cleanup before parsing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        pdf_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, docling_converter.convert, tmp_path)
            doc = result.document

            chunks_gen = chunker.chunk(dl_doc=doc)
            formatted_chunks = []

            for i, chunk in enumerate(chunks_gen):
                pages = sorted(list(set(
                    prov.page_no for item in chunk.meta.doc_items for prov in item.prov if prov.page_no is not None
                )))
                labels = sorted(list(set(
                    str(item.label) for item in chunk.meta.doc_items
                )))

                formatted_chunks.append({
                    "chunk_index": i,
                    "text": chunk.text,
                    "contextualized_text": chunker.contextualize(chunk),
                    "headings": chunk.meta.headings if chunk.meta.headings else [],
                    "pages": pages,
                    "doc_item_labels": labels
                })

            return {
                "total_pages": getattr(doc, "num_pages", 1),
                "chunk_count": len(formatted_chunks),
                "chunks": formatted_chunks
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        logger.error(f"Error during PDF processing: {e}")
        raise HTTPException(status_code=500, detail=f"PDF Processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

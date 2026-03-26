import json

from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import StreamingHttpResponse

from pymongo import MongoClient
from dotenv import load_dotenv
import requests
import os
import  datetime

from .models import Document


#Upload document and process it

from .rag.rag_pipeline import process_document
from .rag.retriever import retrieve_chunks

from .rag.vector_store import stored_chunks
from .rag.summarizer import summarize_document


load_dotenv()

client =MongoClient("mongodb://localhost:27017/")
db=client["ai_chatbot"]
messages_collection = db["messages"]

@api_view(['POST'])
def chat(request):
    message = request.data.get("message")
    message_lower = message.lower()

    is_summary = any(word in message_lower for word in [
        "summarize", "summary", "overview", "brief"
])
    conversation_id = request.data.get("conversation_id", "default")

    messages_collection.insert_one({
        "conversation_id": conversation_id,
        "role": "user",
        "content": message,
        "timestamp": datetime.datetime.utcnow()
    })

    history = list(
        messages_collection.find(
            {"conversation_id": conversation_id}
        )
        .sort("timestamp", -1)
        .limit(20)
    )

    history.reverse()

    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]

      # -----------------------------
    # RAG RETRIEVAL
    # -----------------------------
    try:
            if is_summary:
                print("📄 Running production summarization...")

                if len(stored_chunks) == 0:
                    return Response({"error": "No document indexed"}, status=400)

                summary = summarize_document(stored_chunks)

                # Save assistant response
                messages_collection.insert_one({
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": summary,
                    "timestamp": datetime.datetime.utcnow()
                })

                return Response(summary)

            # ============================================
            # 🔍 NORMAL RAG FLOW
            # ============================================
            relevant_chunks = retrieve_chunks(message)

            if not relevant_chunks:
                context = ""
            else:
                context = "\n".join(relevant_chunks)

            print("RAG CONTEXT:", context)

    except Exception as e:
            print("RAG error:", e)
            context = ""



    api_key = os.getenv("OPENROUTER_API_KEY")

    def generate():

        full_reply = ""

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {"role": "system",  "content": f"""
                    Use the following document context to answer the user question.

                    If the question is about summarizing, provide a structured summary.

                    If the answer is not in the document context, say you don't know.


                    Document Context:
                    {context}

                    If the answer is not in the document context, say you don't know.
                    """},
                    *messages
                ],
                "stream": True
            },
            stream=True
        )
        
                   
        for line in response.iter_lines():

            if not line:
                continue

            decoded = line.decode("utf-8")

            print("STREAM:", decoded)   

            if decoded.startswith("data: "):

                data = decoded[6:]

                if data == "[DONE]":
                    break

                try:
                    # parsed = json.loads(data)
                    # token = parsed["choices"][0]["delta"].get("content", "")
                    parsed = json.loads(data)

                    choices = parsed.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")

                    if token:
                        full_reply += token
                        yield token

                except Exception as e:
                    print("Parse error:", e)

        

                    

        messages_collection.insert_one({
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": full_reply,
            "timestamp": datetime.datetime.utcnow()
        })

    return StreamingHttpResponse(generate(), content_type="text/plain")





@api_view(['POST'])
def upload_document(request):

    file = request.FILES.get("file")
    if not file:
        return Response({"error": "No file uploaded"}, status=400)

    os.makedirs("media/documents", exist_ok=True)
    path = f"media/documents/{file.name}"

    with open(path, "wb+") as f:
        for chunk in file.chunks():
            f.write(chunk)

    print("Processing document:", path)
    process_document(path)
    print("Document indexing finished")
    print("Document processed")
    return Response({"message": "Document processed"})

@api_view(['GET'])

def list_documents(request):
    docs = Document.objects.all().values('id', 'file', 'uploaded_at')
    return Response(docs)




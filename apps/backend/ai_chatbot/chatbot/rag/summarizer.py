import requests
import os
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# 🔥 Better model (recommended)
MODEL = "openai/gpt-4o-mini"


# ============================================
# 🔁 RETRY WRAPPER (Production Safe)
# ============================================
def call_llm(payload, retries=3):
    for attempt in range(retries):
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()

            print(f"⚠️ Retry {attempt+1}: {response.text}")

        except Exception as e:
            print(f"⚠️ Retry error {attempt+1}:", e)

    return None


# ============================================
# 🔹 SUMMARIZE CHUNK (BATCHED)
# ============================================
def summarize_chunk(chunk):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """
Summarize the following text into clear bullet points.

Focus on:
- Key issues
- Important observations
- Problems or insights

Keep it concise and structured.
"""
            },
            {
                "role": "user",
                "content": chunk
            }
        ]
    }

    data = call_llm(payload)

    if not data or "choices" not in data:
        return ""

    return data["choices"][0]["message"]["content"]


# ============================================
# 🔹 BATCH CHUNKS (Reduce API calls)
# ============================================
def batch_chunks(chunks, batch_size=3):
    for i in range(0, len(chunks), batch_size):
        yield "\n\n".join(chunks[i:i+batch_size])


# ============================================
# 🔹 MAP PHASE (Parallel + Batched)
# ============================================
def map_summaries(chunks):
    print(f"🧠 Running MAP on {len(chunks)} chunks...")

    batched_chunks = list(batch_chunks(chunks, batch_size=3))
    summaries = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(summarize_chunk, batched_chunks)

    for res in results:
        if res and res.strip():
            summaries.append(res)

    print(f"✅ MAP completed: {len(summaries)} summaries generated")

    return summaries


# ============================================
# 🔹 CLEAN OUTPUT (Fix formatting issues)
# ============================================
def clean_summary(text):
    text = text.replace("\\n", "\n")
    text = re.sub(r'\s+', ' ', text)   # remove extra spaces
    text = text.replace(" :", ":")
    text = text.strip()
    return text


# ============================================
# 🔹 REDUCE PHASE (Final structured summary)
# ============================================
def reduce_summaries(summaries):
    if not summaries:
        return "No content available to summarize."

    combined = "\n\n".join(summaries)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """
Create a clean, professional summary.

STRICT FORMAT:
- Use proper headings
- Use bullet points under each heading
- Do NOT merge words incorrectly
- Keep spacing clean

Structure:

## Key Issues
- ...

## Design Problems
- ...

## Content Issues
- ...

## Recommendations
- ...

Ensure readability and professional formatting.
"""
            },
            {
                "role": "user",
                "content": combined
            }
        ]
    }

    data = call_llm(payload)

    if not data or "choices" not in data:
        return "Failed to generate summary"

    final_summary = data["choices"][0]["message"]["content"]

    return clean_summary(final_summary)


# ============================================
# 🔹 MAIN PIPELINE
# ============================================
def summarize_document(chunks):
    try:
        print("🚀 Starting summarization pipeline...")

        if not chunks or len(chunks) == 0:
            return "No document content available."

        # 🔥 Smart chunk selection
        if len(chunks) <= 20:
            selected_chunks = chunks
        else:
            selected_chunks = (
                chunks[:8] +
                chunks[len(chunks)//2 - 2 : len(chunks)//2 + 2] +
                chunks[-5:]
            )

        print(f"📄 Selected {len(selected_chunks)} chunks")

        # MAP
        summaries = map_summaries(selected_chunks)

        # REDUCE
        final_summary = reduce_summaries(summaries)

        print("✅ Summarization completed")

        return final_summary

    except Exception as e:
        print("❌ summarize_document error:", e)
        return "Failed to generate summary"
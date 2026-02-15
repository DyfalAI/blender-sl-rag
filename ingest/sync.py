import os
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
import uuid

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# ----- CONFIG -----
SOURCES = [
    {
        "name": "blender-manual",
        "version": "5.0",
        "url": "https://docs.blender.org/manual/en/5.0/"
    }
]

# ------------------

def fetch_page(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def store_document(source, version, path, content, embedding):
    supabase.table("documents").insert({
        "id": str(uuid.uuid4()),
        "source": source,
        "version": version,
        "path": path,
        "content": content,
        "embedding": embedding
    }).execute()

def process_source(source):
    html = fetch_page(source["url"])
    text = extract_text(html)

    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]

    for idx, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        store_document(
            source=source["name"],
            version=source["version"],
            path=f"chunk_{idx}",
            content=chunk,
            embedding=embedding
        )
        print(f"Stored chunk {idx}")

if __name__ == "__main__":
    for source in SOURCES:
        process_source(source)

import os
import glob
import sys
import argparse

# src dizinini ve kök dizini sys.path'e ekle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from database import init_db, reset_db, insert_chunk
from foundry_client import client, EMBEDDING_MODEL

def chunk_text(text, max_chunk_size=500, overlap=50):
    """Metni paragraflara göre böler, gerekirse uzun paragrafları alt parçalara ayırır."""
    paragraphs = text.split('\n\n')
    chunks = []
    for p in paragraphs:
        cleaned_p = p.strip()
        if not cleaned_p:
            continue
        if len(cleaned_p) <= max_chunk_size:
            chunks.append(cleaned_p)
        else:
            # Uzun paragrafı kayan pencere (sliding window) ile alt parçalara böl
            start = 0
            while start < len(cleaned_p):
                end = start + max_chunk_size
                sub_chunk = cleaned_p[start:end].strip()
                if sub_chunk:
                    chunks.append(sub_chunk)
                start += (max_chunk_size - overlap)
    return chunks

def ingest_data(data_dir=None, reset=False):
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, "data")
        
    if reset:
        reset_db()
    else:
        init_db()
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"'{data_dir}' klasörü oluşturuldu. Lütfen içine .txt veya .md dosyaları ekleyip tekrar çalıştırın.")
        return
        
    # .txt ve .md dosyalarını tara
    files = []
    for ext in ("*.txt", "*.md"):
        files.extend(glob.glob(os.path.join(data_dir, ext)))
        
    if not files:
        print(f"'{data_dir}' klasöründe .txt veya .md dosyası bulunamadı.")
        return

    for filepath in files:
        source_name = os.path.basename(filepath)
        print(f"İşleniyor: {source_name} ({filepath})")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
        chunks = chunk_text(text)
        for chunk in chunks:
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=chunk
                )
                embedding_vector = response.data[0].embedding
                insert_chunk(chunk, embedding_vector, source=source_name)
                print(f"  |- Başarıyla eklendi: [{source_name}] {chunk[:35]}...")
            except Exception as e:
                print(f"  |- Embedding alınırken hata oluştu: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veri işleme ve vektör veritabanına aktarma betiği.")
    parser.add_argument("--reset", action="store_true", help="Veritabanını sıfırlayıp yeniden oluşturur.")
    args = parser.parse_args()
    
    print("Veri işleme başlıyor...")
    ingest_data(reset=args.reset)

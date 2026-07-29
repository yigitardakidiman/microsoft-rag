import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from retrieval import get_top_chunks
from foundry_client import client, CHAT_MODEL

def answer_query(user_question):
    print(f"\n🔍 Belgeler aranıyor: '{user_question}'...")
    chunks = get_top_chunks(user_question, top_k=2)
    
    if not chunks:
        return "Üzgünüm, veritabanında ilgili belge bulunamadı veya sunucu bağlantısı başarısız oldu."
        
    print(f"✅ {len(chunks)} adet ilgili belge parçası bulundu.")
    for idx, c in enumerate(chunks, 1):
        print(f"   [{idx}] Kaynak: {c['source']} (Skor: {c['score']:.4f}) -> {c['content'][:50]}...")
        
    formatted_context = "\n\n---\n\n".join([f"[Kaynak: {c['source']}]\n{c['content']}" for c in chunks])
    
    user_prompt = f"""Aşağıdaki BAĞLAM bilgilerini kullanarak soruyu cevapla.
Yanıtının sonuna faydalandığın kaynak adını ekle (örnek: [Kaynak: {chunks[0]['source']}]).
Eğer verilen bağlamda sorunun cevabı yoksa, SADECE "Bu bilgiye sahip değilim." yanıtını ver ve uydurma.

BAĞLAM:
{formatted_context}

SORU:
{user_question}"""

    print(f"🤖 Yapay zeka ({CHAT_MODEL}) yanıt üretiyor...\n")
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Sen yerel bir soru-cevap asistansın. Verilen bağlam dışına çıkma."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            stream=False
        )
        answer = response.choices[0].message.content
        if "<think>" in answer and "</think>" in answer:
            answer = answer.split("</think>")[-1].lstrip()
        elif "<think>" in answer:
            answer = answer.split("<think>")[0].rstrip()
            
        if "SORU:" in answer:
            answer = answer.split("SORU:")[-1].strip()
            if user_question in answer:
                answer = answer.replace(user_question, "").strip()
                
        if not answer.strip():
            answer = "Bu bilgiye sahip değilim."
            
        return answer
    except Exception as e:
        return f"Model hatası: {e}"

def main():
    print("==================================================")
    print("🤖 Yerel RAG Asistanı CLI (Komut Satırı Arayüzü)")
    print("Çıkmak için 'q' veya 'exit' yazabilirsiniz.")
    print("==================================================\n")
    
    while True:
        try:
            user_input = input("Soru > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['q', 'exit', 'cikis', 'çıkış']:
                print("Görüşmek üzere!")
                break
                
            ans = answer_query(user_input)
            print(f"\nCevap:\n{ans}\n")
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nProgram kapatılıyor...")
            break

if __name__ == "__main__":
    main()

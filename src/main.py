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
    
    system_prompt = f"""Sen yardımsever bir asistansın.
Aşağıda verilen bağlam (context) bilgilerini kullanarak kullanıcının sorusunu cevapla.
Cevap verirken faydalandığın kaynak dosya adını (örneğin [Kaynak: sample.txt]) belirt.
Eğer cevabı verilen bağlamda bulamıyorsan 'Bu bilgiye sahip değilim.' de ve uydurma.

BAĞLAM:
{formatted_context}
"""

    print(f"🤖 Yapay zeka ({CHAT_MODEL}) yanıt üretiyor...\n")
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            max_tokens=600,
            stream=False
        )
        answer = response.choices[0].message.content
        # Düşünme etiketlerini temizle
        if "<think>" in answer and "</think>" in answer:
            answer = answer.split("</think>")[-1].lstrip()
        elif "<think>" in answer:
            answer = answer.split("<think>")[0].rstrip()
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

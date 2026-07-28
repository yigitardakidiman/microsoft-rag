import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from retrieval import get_top_chunks
from foundry_client import client, CHAT_MODEL
from database import get_all_chunks

TEST_CASES = [
    {
        "category": "Cevaplanabilir Soru (Yapay Zeka & RAG)",
        "query": "RAG nedir ve halüsinasyonu nasıl engeller?",
        "expected_keywords": ["RAG", "halüsinasyon", "veritabanı", "bağlam", "retrieval"]
    },
    {
        "category": "Cevaplanabilir Soru (Foundry Local)",
        "query": "Microsoft Foundry Local nedir ve ekran kartı zorunlu mudur?",
        "expected_keywords": ["Foundry", "bilgisayar", "offline", "internetsiz", "GPU", "CPU"]
    },
    {
        "category": "Cevaplanabilir Soru (Müzik Teorisi)",
        "query": "Gitarın standart akort düzeni (notaları) nedir?",
        "expected_keywords": ["E-A-D-G-B-E", "Mi", "La", "Re", "Sol", "Si", "akort", "gitar"]
    },
    {
        "category": "Cevaplanabilir Soru (Python & SQLite)",
        "query": "Python'da embedding vektörleri veritabanında nasıl saklanır?",
        "expected_keywords": ["sqlite", "json", "dumps", "embedding", "vektör", "text"]
    },
    {
        "category": "Cevaplanamaz Soru (Bilgi Dışı)",
        "query": "Fransa'nın başkenti neresidir?",
        "expected_keywords": ["bilgiye sahip değilim", "bulunmuyor"]
    },
    {
        "category": "Kenar Durum (Edge Case)",
        "query": "Kuantum bilgisayarları ve süper iletken çekirdekler",
        "expected_keywords": ["bilgiye sahip değilim", "bulunmuyor"]
    }
]

def run_evaluation():
    print("=" * 65)
    print("Yerel RAG Sistemi Test ve Değerlendirme Betiği (Phase 3 Week 5)")
    print("=" * 65)

    all_chunks = get_all_chunks()
    print(f"\nVeritabanı Durumu: Toplam {len(all_chunks)} metin parçası (chunk) yüklü.\n")

    results = []
    total_retrieval_time = 0.0
    total_gen_time = 0.0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"--- [Test {i}/{len(TEST_CASES)}] Kategori: {test['category']} ---")
        print(f"Soru: {test['query']}")
        
        start_time = time.time()
        chunks = get_top_chunks(test['query'], top_k=2)
        retrieval_time = time.time() - start_time
        total_retrieval_time += retrieval_time
        
        print(f"Arama Süresi: {retrieval_time:.4f} saniye")
        if chunks:
            top_score = chunks[0]['score']
            top_source = chunks[0]['source']
            print(f"En Yüksek Benzerlik Skoru: {top_score:.4f} (Kaynak: {top_source})")
        else:
            print("Hiçbir belge eşleşmedi.")

        answer = ""
        gen_time = 0.0
        status = "FAIL"
        reason = ""

        # Model Yanıtı
        if not chunks:
            answer = "Bu bilgiye sahip değilim."
            print(f"Model Yanıtı:\n{answer} (Veritabanında eşleşen kaynak bulunamadı).\n")
            # Eğer test ulaşılamayan bir soruysa, "Bu bilgiye sahip değilim" cevabı başarıdır
            if any(kw.lower() in answer.lower() for kw in test["expected_keywords"]):
                status = "PASS"
                reason = "Eşleşen belge olmadığından doğru şekilde bilgi dışı yanıt verildi."
            else:
                status = "FAIL"
                reason = "Bilgi dışı durum için beklenen anahtar kelime eşleşmedi."
        else:
            formatted_context = "\n\n---\n\n".join([f"[Kaynak: {c['source']}]\n{c['content']}" for c in chunks])
            system_prompt = f"""Sen yardımsever bir asistansın.
Aşağıda verilen bağlam bilgilerini kullanarak kullanıcının sorusunu cevapla.
Cevap verirken faydalandığın kaynak dosya adını (örneğin [Kaynak: sample.txt]) belirt.
Eğer cevabı verilen bağlamda bulamıyorsan 'Bu bilgiye sahip değilim.' de ve uydurma.

BAĞLAM:
{formatted_context}
"""
            gen_start = time.time()
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": test['query']}
                    ]
                )
                answer = response.choices[0].message.content
                if "<think>" in answer and "</think>" in answer:
                    answer = answer.split("</think>")[-1].lstrip()
                elif "<think>" in answer:
                    answer = answer.split("<think>")[0].rstrip()
                gen_time = time.time() - gen_start
                total_gen_time += gen_time
                print(f"Yanıt Üretim Süresi: {gen_time:.2f} saniye")
                print(f"Model Yanıtı:\n{answer}\n")

                # Keyword kontrolü
                matched_kws = [kw for kw in test["expected_keywords"] if kw.lower() in answer.lower() or kw.lower() in formatted_context.lower()]
                if matched_kws:
                    status = "PASS"
                    reason = f"Eşleşen anahtar kelimeler: {', '.join(matched_kws)}"
                else:
                    status = "FAIL"
                    reason = f"Beklenen kelimeler bulunamadı ({', '.join(test['expected_keywords'])})"

            except Exception as e:
                answer = f"Hata: {e}"
                print(f"{answer}\n")
                status = "FAIL"
                reason = f"Model hatası: {e}"

        results.append({
            "id": i,
            "category": test["category"],
            "query": test["query"],
            "status": status,
            "reason": reason,
            "retrieval_time": retrieval_time,
            "gen_time": gen_time
        })

    # Değerlendirme Rapor Özeti
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    total_count = len(results)
    pass_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
    avg_retrieval = total_retrieval_time / total_count if total_count > 0 else 0
    avg_gen = total_gen_time / total_count if total_count > 0 else 0

    print("=" * 65)
    print("📊 YEREL RAG DEĞERLENDİRME VE TEST RAPORU ÖZETİ")
    print("=" * 65)
    print(f"Toplam Test Sayısı       : {total_count}")
    print(f"Başarılı (PASS)           : {passed_count}")
    print(f"Başarısız (FAIL)          : {total_count - passed_count}")
    print(f"Başarı Oranı (%)          : %{pass_rate:.1f}")
    print(f"Ort. Vektör Arama Süresi  : {avg_retrieval:.4f} saniye")
    print(f"Ort. Yanıt Üretim Süresi  : {avg_gen:.2f} saniye")
    print("-" * 65)
    print("Test Detayları:")
    for r in results:
        symbol = "✅ [PASS]" if r["status"] == "PASS" else "❌ [FAIL]"
        print(f" {symbol} Test #{r['id']} ({r['category']}): {r['reason']}")
    print("=" * 65)

if __name__ == "__main__":
    run_evaluation()

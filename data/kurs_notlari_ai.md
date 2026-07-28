# Yapay Zeka ve RAG (Retrieval-Augmented Generation) Kurs Notları

## 1. RAG (Retrieval-Augmented Generation) Nedir?
RAG, büyük dil modellerinin (LLM) harici bilgi kaynaklarından yararlanarak daha doğru, güncel ve kaynak doğrulamalı yanıtlar üretmesini sağlanan bir yapay zeka mimarisidir.

### RAG'in Temel Bileşenleri:
- **Retrieval (Arama/Getirme):** Kullanıcı sorusuyla ilişkili belgelerin vektör veritabanından bulunması.
- **Augmentation (Zenginleştirme):** Çekilen ilgili belgenin, kullanıcı sorusuyla birlikte dil modeline yönlendirilecek isteme (prompt) eklenmesi.
- **Generation (Üretim):** Dil modelinin sağlanan bağlama (context) sadık kalarak yanıt üretmesi.

---

## 2. Metin Gömme (Embeddings) ve Benzerlik Arama
Metin gömme (embedding), kelimeleri veya metin parçalarını yüksek boyutlu bir vektör uzayında sayısal dizilere dönüştürme işlemidir. Anlamsal olarak birbirine benzeyen metinler, vektör uzayında birbirine yakın noktalarda yer alır.

### Kosinüs Benzerliği (Cosine Similarity):
İki vektör arasındaki açının kosinüsünü hesaplayarak aralarındaki anlamsal benzerliği 0 ile 1 arasında ölçeklendirir. 1'e yakın değerler yüksek anlamsal benzerliği gösterir.

---

## 3. Sorumlu Yapay Zeka ve Halüsinasyonu Önleme
Dil modelleri, eğitildikleri veride bulunmayan konularda mantıklı görünen ancak yanlış bilgiler (halüsinasyon) üretebilir. RAG mimarisinde sistem promptuna *"Sadece verilen bağlamı kullan, cevabı bulamazsan bilmiyorum de"* kuralı eklenerek halüsinasyon riski ortadan kaldırılır.

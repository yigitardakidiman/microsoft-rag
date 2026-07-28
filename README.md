# 🤖 Microsoft Foundry Local ile Yerel RAG Yapay Zeka Asistanı

Bu proje, **Microsoft Foundry Local** ve **RAG (Retrieval-Augmented Generation)** kalıbı kullanılarak geliştirilmiş, tamamen internetsiz (offline) çalışan yerel bir soru-cevap asistanıdır.

Kullanıcıların kendi belgelerinden (örneğin `.txt` formatındaki ders notları, kılavuzlar veya SSS dokümanları) bilgi çekerek, bulut servislerine ihtiyaç duymadan cihaz üzerinde çalışan yapay zeka modelleriyle doğru ve kaynak atıflı (cited) yanıtlar üretmesini sağlar.

---

## 📐 Sistem Mimarisi ve Proje Dizin Yapısı

Projenin tüm kaynak kodları **`src/`** klasörü altında modüler bir yapıda organize edilmiştir:

```
microsoft-rag/
├── 📄 README.md                 # Proje ana dokümantasyonu
├── 📄 requirements.txt           # Gerekli bağımlılıklar
├── 📁 src/                      # Kaynak Kod Klasörü
│   ├── ⚙️ foundry_client.py     # Foundry Local API istemci bağlantısı
│   ├── 🗄️ database.py           # SQLite veritabanı ve metadata yönetimi
│   ├── 🔄 ingest.py             # Metin parçalama ve vektörleştirme
│   ├── 🔍 retrieval.py          # Kosinüs benzerliği ile vektör arama (Threshold: 0.45)
│   ├── 🖥️ app.py                # Streamlit Web Arayüzü
│   ├── 🖥️ main.py               # Terminal CLI Arayüzü
│   └── 🧪 test_rag.py           # Otomatik test ve değerlendirme betiği
├── 📁 data/                     # RAG için kaynak .txt belgeleri
│   └── 📄 sample.txt
└── 📁 docs/                     # Müfredat ve plan belgeleri
    └── 📄 Summer School Foundry Local Plan.pdf
```

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r requirements.txt
```

### 2. Foundry Local Sunucusunun Başlatılması
Foundry yerel sunucusunu başlatın ve modelleri yükleyin:
```bash
foundry server start -p 64390
foundry model load qwen3-embedding-0.6b-generic-cpu:1
foundry model load qwen3-0.6b-generic-cpu:4
```

### 3. Belgelerin İşlenmesi (Veri Yükleme)
```bash
# Belgeleri veritabanına aktarmak için:
python src/ingest.py --reset
```

### 4. Asistanı Başlatma

#### Option A: Web Arayüzü (Streamlit)
```bash
streamlit run src/app.py
```

#### Option B: Terminal CLI Arayüzü
```bash
python src/main.py
```

### 5. Test ve Değerlendirme
```bash
python src/test_rag.py
```
from openai import OpenAI

# Foundry Local masaüstü uygulaması genellikle 8080 veya 5272 gibi portlardan yayın yapar.
# Varsayılan adres: http://127.0.0.1:64390/v1
base_url = "http://127.0.0.1:64390/v1" 

print(f"Yapay zeka sunucusuna bağlanılıyor: {base_url}")

# Doğrudan OpenAI kütüphanesi ile yerel sunucuya bağlanıyoruz
client = OpenAI(
    base_url=base_url,
    api_key="local-key", # Yerel araçlar genelde api key sormaz
    timeout=300.0 # CPU yükü ve uzun yanıtlar için 300sn (5 dk) zaman aşımı
)

# Foundry Local içerisinden indirdiğiniz model isimleri
EMBEDDING_MODEL = "qwen3-embedding-0.6b-generic-cpu:1"
CHAT_MODEL = "qwen3-0.6b-generic-cpu:4"

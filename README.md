# Rag Project

😠 Angry RAG Chatbot
Bu proje, PDF dosyalarından belge içeriğini okuyup bölerek vektör veritabanında saklayan ve kullanıcı sorularına bu belgelerden bilgi çekerek öfkeli cevaplar veren bir RAG (Retrieval-Augmented Generation) uygulamasıdır.

🚀 Özellikler
📄 PDF klasöründen belge okuma ve bölme
🧠 OpenAI Embedding kullanarak vektör veritabanı oluşturma
🤖 GPT-4o-mini ile bilgiye dayalı cevaplar üretme
🗂️ Her chunk için doc_type ve page_number metadata desteği
😠 Sorulara "sinirli" yapay zeka patron tonu ile yanıt verme

📚 Kaynak gösterimi: yanıtın dayandığı sayfa ve belge türü bilgisi

## Klasör Yapısı

```
├── app.py              # FastAPI API ucu
├── rag_project.py      # Ana sınıf ve RAG işlemleri
├── data/               # PDF belgelerinin bulunduğu klasör
├── rag_db/             # Chroma vektör veritabanı
└── README.md           # Bu dosya
```

data/ klasörüne PDF dosyalarını yerleştir.
API'yi çalıştır:
```
uvicorn app:app --reload
```

Post isteği ile soru gönder:
```
json
POST /ask
{
    "question": "Raporlarda geçen temel maddeler nelerdir?"
}
```

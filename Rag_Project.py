import uvicorn
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

class RagProject:
    def __init__(self):


        self.model = "gpt-4o-mini"

        self.openai = OpenAI()
        self.documents = "data/"

        self.memory = None
        self.retriever = None
        self.llm = None
        self.conversation_chain = None
        self.vectorstore = None

        self.system_message = """ 
        You are an angry ai boss. Answer the user's question honestly and clearly, based solely on information 
        found in the documentation. If the documentation does not answer, say "I don't know the answer to that 
        question." angry also even if you don't know the answer. Each answer should be angry tone
        """

        self.prompt = self.build_prompt()

    def build_prompt(self):
        return ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(self.system_message),
    HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion:\n{question}")
    ])


    def load_documents(self):

        loader = PyPDFDirectoryLoader(self.documents)
        return loader.load()

    def split_documents(self):

        docs = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        return text_splitter.split_documents(docs)

    def create_embedding(self):

        db_name = "rag_db"
        embeddings = OpenAIEmbeddings()

        if os.path.exists(db_name) and os.listdir(db_name):
            self.vectorstore = Chroma(
                persist_directory=db_name,
                embedding_function=embeddings
            )
        else:
            chunks = self.split_documents()
            for chunk in chunks:
                chunk.metadata["doc_type"] = "report"
                chunk.metadata["page_number"] = chunk.metadata.get("page", None)
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_name
            )


    def setup_llm(self):

        self.create_embedding()

        self.llm = ChatOpenAI(model_name=self.model, temperature=0.7)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def rag_chain(self):
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},

        )

    def response_with_sources(self, question: str):
        chain = self.rag_chain()
        response = chain.invoke({"question": question})

        answer = response["answer"]
        sources = response.get("source_documents", [])

        if not sources:
            return f"😠 {answer.strip()}\n\n📚 **Kaynaklar:**\nKaynak bulunamadı."

        source_info = "\n\n📚 **Kaynaklar:**\n"
        for i, doc in enumerate(sources):
            page = doc.metadata.get("page_number", "Bilinmiyor")
            doc_type = doc.metadata.get("doc_type", "Bilinmiyor")
            snippet = doc.page_content[:200].strip().replace("\n", " ")
            source_info += f"\n{i + 1}. 📄 Belge Türü: {doc_type} | Sayfa: {page}\n   ➤ “{snippet}...”"

        return f"😠 {answer.strip()}\n{source_info}"


rag_utils = RagProject()
rag_utils.setup_llm()
rag_chain = rag_utils.rag_chain()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    try:
        print("Soru alındı:", q.question)
        response = rag_utils.response_with_sources(q.question)
        print("Cevap döndü:", response)
        return {"Answer": response}
    except Exception as e:
        print("Bir hata oluştu:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


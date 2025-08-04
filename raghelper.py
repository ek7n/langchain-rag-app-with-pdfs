import os
from dotenv import load_dotenv
from typing import List, Tuple
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

my_key_openai = os.getenv("openai_apikey")
my_key_google = os.getenv("google_apikey")

llm_gemini = ChatGoogleGenerativeAI(google_api_key=my_key_google, model="gemini-2.5-flash")
#embeddings = OpenAIEmbeddings(api_key=my_key_openai)
# birden fazla dilde kaynak belgeden döküman getirme için OpenAI'ya göre daha başarılıydı.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")



def create_vectorstore_from_pdfs(pdf_folder_path: str) -> FAISS:
    if not os.path.exists(pdf_folder_path):
        raise FileNotFoundError(f"PDF klasörü bulunamadı: {pdf_folder_path}")

    all_raw_docs = []
    pdf_files = Path(pdf_folder_path).glob("*.pdf")

    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(file_path=str(pdf_path))
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["source"] = pdf_path.name
        all_raw_docs.extend(raw_docs)

    if not all_raw_docs:
        raise ValueError("Klasörde PDF bulunamadı veya içerik çıkarılamadı.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    chunks = splitter.split_documents(all_raw_docs)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    from collections import Counter
    #print("\n Her PDF için oluşturulan parça (chunk) sayısı:")
    counts = Counter(chunk.metadata["source"] for chunk in chunks)
    for source, count in counts.items():
        print(f"→ {source}: {count} parça")
    if not chunks:
        raise ValueError("Parçalama sonrası anlamlı içerik bulunamadı.")

    vectorstore = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
    return vectorstore


def rag_with_pdf(vectorstore: FAISS, prompt: str, k: int = 4) -> Tuple[str, List[Document]]:
    retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': k, 'fetch_k': 20})
    retriever_multi_query_mmr = MultiQueryRetriever.from_llm(
        retriever=retriever_mmr, llm=llm_gemini
    )
    relevant = retriever_multi_query_mmr.invoke(prompt)

    context = " ".join(d.page_content for d in relevant)
    system_prompt = f"""
    Sen, yaşlanma, yaşlılık politikaları, sağlıklı yaşam ve aktif yaşlanma konularında uzmanlaşmış bir araştırma asistanısın.

    Kullanıcının sorusu:
    "{prompt}"

    Aşağıdaki belgeler, bu soruyu yanıtlamanda tek bilgi kaynağındır:
    {context}

    Kurallar:
    - Yanıtlarını yalnızca bu belgelere dayandır.
    - Eğer doğrudan cevap yoksa, belgelerden çıkarılabilecek bilgileri kullanarak açıklayıcı bir yanıt üret.
    - Uydurma bilgi verme, ancak yorumlama yapmaktan çekinme.
    - Açık, akademik ama sade bir dille yanıtla.
    """

    answer = llm_gemini.invoke(system_prompt).content
    return answer, relevant


def get_retrieved_metadata(docs: List[Document]) -> str:
    metadata_list = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Bilinmiyor')
        page_info = doc.metadata.get('page', 'Bilinmiyor')
        preview = doc.page_content[:300].strip().replace('\n', ' ')
        metadata_list.append(f"**Belge {i + 1} (Kaynak: {source}, Sayfa: {page_info}):**\n{preview}...")
    return "\n\n---\n\n".join(metadata_list)

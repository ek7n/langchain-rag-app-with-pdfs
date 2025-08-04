import streamlit as st
import streamlit.components.v1 as components
import raghelper
from pathlib import Path
import traceback
import os

st.set_page_config(page_title="LangChain ile Bellek Genişletme", layout="wide")
st.title("LangChain ile Bellek Genişletme: Yaşlanma Raporları")
st.divider()

# --- Otomatik yumuşak kaydırma için JS ---
components.html(
    """
    <script>
        const observer = new MutationObserver(() => {
            const target = document.getElementById("scroll-target");
            if (target) {
                target.scrollIntoView({ behavior: "smooth" });
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """,
    height=0
)

# --- Scroll hedef noktası ---
st.markdown('<div id="scroll-target"></div>', unsafe_allow_html=True)

# PDF klasör yolu ve sabit k değeri
pdf_folder_path = "data"
k_value = 5  # Sabit alınacak döküman sayısı

# --- Sidebar bilgi amaçlı ---
with st.sidebar:
    st.write("Uygulamanın çalışması için gerekli PDF'ler, ana dizindeki `data` klasöründe bulunmalıdır.")
    st.markdown("---")

# --- Vektör deposunu önbelleğe alan fonksiyon ---
@st.cache_resource
def load_and_cache_vectorstore(path_to_folder: str):
    with st.spinner("PDF'ler yükleniyor ve vektör deposu hazırlanıyor... Bu biraz zaman alabilir."):
        try:
            vectorstore = raghelper.create_vectorstore_from_pdfs(path_to_folder)
            st.success("Tüm PDF'ler başarıyla yüklendi ve RAG sistemi hazır!")
            return vectorstore
        except Exception as e:
            st.error(f"PDF'ler yüklenirken kritik hata oluştu: {e}")
            st.text(traceback.format_exc())
            return None

# Vektör deposunu yükle
vectorstore = load_and_cache_vectorstore(pdf_folder_path)

if vectorstore is None:
    st.error("RAG sistemi başlatılamadı.")
    st.stop()

if not os.path.exists(pdf_folder_path):
    st.error(f"Hata: Belirtilen PDF klasörü bulunamadı: {pdf_folder_path}")
    st.stop()

# Sohbet geçmişini başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sohbet Geçmişini Görüntüleme ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message and message["metadata"]:
            with st.expander("Kaynak Belgeleri"):
                st.write(message["metadata"])

# --- Kullanıcıdan Giriş Alma ---
if prompt := st.chat_input("Sorunuzu buraya yazın:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    rag_answer = ""
    rag_metadata = ""

    with st.spinner("RAG ile yanıt aranıyor..."):
        try:
            answer, docs = raghelper.rag_with_pdf(vectorstore, prompt, k=k_value)
            rag_answer = answer

            metadata_list = []
            if docs:
                for i, doc in enumerate(docs):
                    source = doc.metadata.get('source', 'Bilinmiyor')
                    page_info = doc.metadata.get('page', 'Bilinmiyor')
                    metadata_list.append(
                        f"**Belge {i + 1} (Kaynak: {source}, Sayfa: {page_info}):**\n{doc.page_content[:300]}..."
                    )
            rag_metadata = "\n\n---\n\n".join(metadata_list)

        except Exception as e:
            rag_answer = f"RAG çağrısında hata oluştu: {e!r}"
            st.error(rag_answer)
            st.text(traceback.format_exc())

    with st.chat_message("assistant"):
        st.markdown(rag_answer)
        if rag_metadata:
            st.expander("Kaynak Belgeleri").write(rag_metadata)

    st.session_state.messages.append(
        {"role": "assistant", "content": f"### Yanıt (RAG):\n{rag_answer}", "metadata": rag_metadata}
    )

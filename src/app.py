import os
import sys
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from retrieval import get_top_chunks
from foundry_client import client, CHAT_MODEL

st.set_page_config(page_title="Yerel RAG Asistanı", layout="centered", page_icon="🤖")

st.title("🤖 Yerel RAG Yapay Zeka Asistanı")
st.markdown("Belgelerinizdeki bilgilere dayanarak internet olmadan sorularınızı yanıtlar ve kaynak gösterir.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "retrieved_chunks" in msg and msg["retrieved_chunks"]:
            with st.expander("📚 Getirilen Belgeler ve Kaynaklar"):
                for idx, chunk in enumerate(msg["retrieved_chunks"], 1):
                    st.markdown(f"**Parça {idx}** | *Kaynak:* `{chunk['source']}` | *Benzerlik Skoru:* `{chunk['score']:.4f}`")
                    st.caption(chunk["content"])

if prompt := st.chat_input("Belgelerle ilgili bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Belgeler içinde aranıyor..."):
            relevant_chunks = get_top_chunks(prompt, top_k=2)
            
        if not relevant_chunks:
            response = "Veritabanında uygun belge bulamadım veya Foundry sunucusuna bağlanamadım."
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # Transparanlık için getirilen belgeleri göster
            with st.expander("📚 Getirilen Belgeler ve Kaynaklar", expanded=True):
                for idx, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Parça {idx}** | *Kaynak:* `{chunk['source']}` | *Benzerlik Skoru:* `{chunk['score']:.4f}`")
                    st.caption(chunk["content"])
            
            formatted_context_list = []
            for chunk in relevant_chunks:
                formatted_context_list.append(f"[Kaynak: {chunk['source']}]\n{chunk['content']}")
            
            context = "\n\n---\n\n".join(formatted_context_list)
            
            user_prompt = f"""Aşağıdaki BAĞLAM bilgilerini kullanarak soruyu cevapla.
Yanıtının sonuna faydalandığın kaynak adını ekle (örnek: [Kaynak: {relevant_chunks[0]['source']}]).
Eğer verilen bağlamda sorunun cevabı yoksa, SADECE "Bu bilgiye sahip değilim." yanıtını ver ve uydurma.

BAĞLAM:
{context}

SORU:
{prompt}"""

            with st.spinner(f"Cevap üretiliyor... (Model: {CHAT_MODEL})"):
                try:
                    stream = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=[
                            {"role": "system", "content": "Sen yerel bir soru-cevap asistansın. Verilen bağlam dışına çıkma."},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=600,
                        stream=True
                    )
                    full_response = ""
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            
                            # Düşünme kısımlarını ve prompt sızıntılarını temizle
                            display_text = full_response
                            if "<think>" in display_text:
                                if "</think>" in display_text:
                                    display_text = display_text.split("</think>")[-1].lstrip()
                                else:
                                    display_text = "💭 *Yapay zeka düşünüyor ve yanıtı hazırlıyor...*"
                            
                            message_placeholder.markdown(display_text + " ▌")
                            
                    # Stream bittikten sonra temizle
                    final_display = full_response
                    if "<think>" in final_display and "</think>" in final_display:
                        final_display = final_display.split("</think>")[-1].lstrip()
                    elif "<think>" in final_display:
                        final_display = final_display.split("<think>")[0].rstrip()

                    # Eğer model prompt metnini sızdırdıysa temizle
                    if "SORU:" in final_display:
                        final_display = final_display.split("SORU:")[-1].strip()
                        # Eğer soru metnini de içeriyorsa soruyu çıkar
                        if prompt in final_display:
                            final_display = final_display.replace(prompt, "").strip()

                    if not final_display.strip() or final_display.strip() == prompt:
                        final_display = "Bu bilgiye sahip değilim."

                    message_placeholder.markdown(final_display)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_display,
                        "retrieved_chunks": relevant_chunks
                    })
                except Exception as e:
                    response = f"Yapay zeka ile iletişimde hata oluştu: {e}"
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

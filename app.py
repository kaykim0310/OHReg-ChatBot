"""
🏭 안전환경 법규 AI 상담사
빠른 로딩 버전 (압축 데이터 사용)
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pickle
import gzip
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="안전환경 법규 AI 상담사",
    page_icon="🏭",
    layout="centered"
)

# ============================================================
# 나눔고딕 폰트
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Nanum Gothic', sans-serif; }
h1, h2, h3 { font-family: 'Nanum Gothic', sans-serif; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# API 키
# ============================================================
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]

# ============================================================
# 세션 상태
# ============================================================
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ============================================================
# 데이터 로드 (캐시됨)
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

@st.cache_resource
def load_data_and_build_db():
    """압축된 데이터 로드 + 벡터 DB 구축"""
    
    # 1. 압축 데이터 로드
    with gzip.open('law_data.pkl.gz', 'rb') as f:
        all_data = pickle.load(f)
    
    # 2. ChromaDB 구축
    chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))
    
    try:
        chroma_client.delete_collection("osh_law")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="osh_law",
        metadata={"hnsw:space": "cosine"}
    )
    
    # 3. 배치로 추가
    batch_size = 100
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i+batch_size]
        
        collection.add(
            documents=[item['full_text'][:1500] for item in batch],
            embeddings=[item['embedding'] for item in batch],
            metadatas=[{
                "type": str(item['type']),
                "law_name": str(item['law_name']),
                "number": str(item['number']),
                "title": str(item.get('title', ''))
            } for item in batch],
            ids=[f"item_{i+j}" for j in range(len(batch))]
        )
    
    return all_data, collection

def search_law(query, collection, embedding_model, n_results=5):
    query_embedding = embedding_model.encode(query).tolist()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

def ask_chatbot(question, collection, embedding_model):
    search_results = search_law(question, collection, embedding_model)
    context = "\n\n---\n\n".join(search_results['documents'][0])
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    prompt = f"""당신은 안전환경 법규 전문 상담사입니다.
아래 제공된 법령, 고시, 별표를 참고하여 질문에 답변해주세요.

## 참고 자료:
{context}

## 질문:
{question}

## 답변 지침:
1. 반드시 위 자료 내용을 근거로 답변하세요
2. 출처를 명확히 밝히세요 (예: "산업안전보건법 제29조에 따르면...")
3. 자료에 없는 내용은 "해당 내용은 제공된 자료에서 찾지 못했습니다"라고 답하세요
4. 쉽고 친절하게 설명하세요
5. 마지막에 면책조항: "※ 본 답변은 참고용이며, 정확한 법률 해석은 전문가와 상담하시기 바랍니다."

## 답변:"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text, search_results

# ============================================================
# 메인 UI
# ============================================================
st.title("🏭 안전환경 법규 AI 상담사")
st.markdown("산업안전, 화학물질, 환경, 위험물 관련 법규를 물어보세요!")

# 사이드바
with st.sidebar:
    st.header("📚 포함된 법규 (33개)")
    
    with st.expander("🔧 산업안전보건 (4개)"):
        st.markdown("산업안전보건법, 시행령, 시행규칙, 안전보건기준규칙")
    
    with st.expander("🧪 화학물질 (6개)"):
        st.markdown("화학물질관리법, 화평법 + 각 시행령/규칙")
    
    with st.expander("🔥 위험물/고압가스 (6개)"):
        st.markdown("위험물안전관리법, 고압가스법 + 각 시행령/규칙")
    
    with st.expander("🌿 환경 (9개)"):
        st.markdown("대기환경보전법, 물환경보전법, 폐기물관리법 + 각 시행령/규칙")
    
    with st.expander("📋 고시/지침 (8개)"):
        st.markdown("위험성평가, MSDS, 노출기준, 작업환경측정, 사무실공기, 위험물세부, 고압가스, 유해화학물질취급")
    
    st.markdown("---")
    st.header("💡 질문 예시")
    st.markdown("""
    - 안전관리자 선임 기준은?
    - MSDS 작성 방법은?
    - 위험물 저장소 기준은?
    - 대기배출허용기준은?
    """)
    
    st.markdown("---")
    st.markdown("⚠️ 본 서비스는 참고용입니다.")
    st.markdown("Made with ❤️ by 힐스")

# 시스템 초기화
with st.spinner("🔄 시스템 준비 중..."):
    embedding_model = load_embedding_model()
    all_data, collection = load_data_and_build_db()

# 통계
article_count = len([d for d in all_data if d['type'] == '조문'])
table_count = len([d for d in all_data if d['type'] == '별표'])
st.success(f"✅ 준비 완료! (조문 {article_count:,}개 + 별표 {table_count}개)")

# 채팅 히스토리
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 검색 중..."):
            try:
                answer, search_results = ask_chatbot(prompt, collection, embedding_model)
                st.markdown(answer)
                
                with st.expander("📜 참고 자료 보기"):
                    for doc, meta in zip(search_results['documents'][0], search_results['metadatas'][0]):
                        badge = "📋" if meta['type'] == '조문' else "📊"
                        st.markdown(f"**{badge} {meta['law_name']} {meta['number']}**")
                        st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                        st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"오류: {str(e)}")

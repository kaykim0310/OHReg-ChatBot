"""
🏭 안전환경 법규 AI 상담사
하이브리드 검색 버전 (벡터 + 키워드)
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
    
    with gzip.open('law_data.pkl.gz', 'rb') as f:
        all_data = pickle.load(f)
    
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
                "title": str(item.get('title', '')),
                "idx": str(i+j)
            } for j, item in enumerate(batch)],
            ids=[f"item_{i+j}" for j in range(len(batch))]
        )
    
    return all_data, collection

def keyword_search(query, all_data, max_results=5):
    """키워드 기반 검색 - 제목과 내용에서 검색"""
    results = []
    query_lower = query.lower()
    
    # 핵심 키워드 추출
    keywords = []
    if '작업환경측정' in query:
        keywords.append('작업환경측정')
    if '안전관리자' in query:
        keywords.append('안전관리자')
    if '보건관리자' in query:
        keywords.append('보건관리자')
    if '안전보건교육' in query or '교육' in query:
        keywords.append('안전보건교육')
    if '과태료' in query:
        keywords.append('과태료')
    if '유해인자' in query or '대상물질' in query:
        keywords.append('유해인자')
    if 'msds' in query_lower or '물질안전보건자료' in query:
        keywords.append('물질안전보건자료')
    if '위험물' in query:
        keywords.append('위험물')
    if '특수건강진단' in query or '건강진단' in query:
        keywords.append('특수건강진단')
    if '노출기준' in query:
        keywords.append('노출기준')
    if '허용기준' in query:
        keywords.append('허용기준')
    if '배출허용' in query or '배출기준' in query:
        keywords.append('배출허용')
    
    # 키워드로 검색
    for idx, item in enumerate(all_data):
        title = item.get('title', '').lower()
        text = item['full_text'][:500].lower()
        
        for kw in keywords:
            if kw.lower() in title or kw.lower() in text:
                results.append({
                    'idx': idx,
                    'item': item,
                    'match': kw
                })
                break
    
    # 별표 우선 정렬
    results.sort(key=lambda x: (0 if x['item']['type'] == '별표' else 1))
    
    return results[:max_results]

def search_law(query, collection, embedding_model, n_results=10):
    """벡터 검색"""
    query_embedding = embedding_model.encode(query).tolist()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

def get_full_text(all_data, idx, max_len=8000):
    """원본 전체 텍스트 가져오기"""
    if 0 <= idx < len(all_data):
        full_text = all_data[idx]['full_text']
        if len(full_text) > max_len:
            return full_text[:max_len] + f"\n\n... (전체 {len(full_text)}자 중 {max_len}자만 표시)"
        return full_text
    return None

def ask_chatbot(question, collection, embedding_model, all_data):
    # 1. 키워드 검색 먼저!
    keyword_results = keyword_search(question, all_data, max_results=3)
    
    # 2. 벡터 검색
    vector_results = search_law(question, collection, embedding_model, n_results=10)
    
    # 3. 결과 병합 (키워드 결과 우선)
    context_parts = []
    selected_metas = []
    used_idx = set()
    
    # 키워드 검색 결과 추가 (별표 전체 텍스트)
    for r in keyword_results:
        idx = r['idx']
        if idx not in used_idx:
            item = r['item']
            full_text = get_full_text(all_data, idx, max_len=10000)
            if full_text:
                context_parts.append(full_text)
                selected_metas.append({
                    'type': item['type'],
                    'law_name': item['law_name'],
                    'number': item['number'],
                    'title': item.get('title', ''),
                    'idx': str(idx)
                })
                used_idx.add(idx)
    
    # 벡터 검색 결과 추가
    for doc, meta in zip(vector_results['documents'][0], vector_results['metadatas'][0]):
        idx = int(meta.get('idx', -1))
        if idx not in used_idx and len(context_parts) < 8:
            max_len = 6000 if meta['type'] == '별표' else 2000
            full_text = get_full_text(all_data, idx, max_len=max_len)
            if full_text:
                context_parts.append(full_text)
                selected_metas.append(meta)
                used_idx.add(idx)
    
    context = "\n\n---\n\n".join(context_parts)
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    prompt = f"""당신은 안전환경 법규 전문 상담사입니다.
아래 제공된 법령 조문과 별표를 참고하여 질문에 답변해주세요.

## 참고 자료:
{context}

## 질문:
{question}

## 답변 지침:
1. 반드시 위 자료 내용을 근거로 답변하세요
2. 출처를 명확히 밝히세요 (예: "산업안전보건법 시행규칙 별표 21에 따르면...")
3. **별표에 목록이나 기준이 있으면 해당 내용을 상세히 인용하세요**
4. **물질 목록을 물어보면 전체 목록을 빠짐없이 나열하세요**
5. 자료에 없는 내용은 "해당 내용은 제공된 자료에서 찾지 못했습니다"라고 답하세요
6. 쉽고 친절하게 설명하세요
7. 마지막에 면책조항: "※ 본 답변은 참고용이며, 정확한 법률 해석은 전문가와 상담하시기 바랍니다."

## 답변:"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return_results = {
        'documents': [context_parts],
        'metadatas': [selected_metas]
    }
    
    return response.content[0].text, return_results

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
    - **작업환경측정대상물질 목록은?**
    - 위험물 저장소 기준은?
    - 안전보건교육 시간은?
    - 과태료 기준이 어떻게 되나요?
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
st.success(f"✅ 준비 완료! (조문 {article_count:,}개 + 별표 {table_count:,}개)")

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
                answer, search_results = ask_chatbot(prompt, collection, embedding_model, all_data)
                st.markdown(answer)
                
                with st.expander("📜 참고 자료 보기"):
                    for doc, meta in zip(search_results['documents'][0], search_results['metadatas'][0]):
                        badge = "📋" if meta['type'] == '조문' else "📊"
                        st.markdown(f"**{badge} {meta['law_name']} {meta['number']}** - {meta['title']}")
                        st.text(doc[:800] + "..." if len(doc) > 800 else doc)
                        st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"오류: {str(e)}")

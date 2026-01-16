"""
🏭 산업안전보건법 AI 상담사
Streamlit 웹 앱
"""

import streamlit as st
import requests
import xml.etree.ElementTree as ET
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="산업안전보건법 AI 상담사",
    page_icon="🏭",
    layout="centered"
)

# ============================================================
# 세션 상태 초기화
# ============================================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ============================================================
# 법령 데이터 로드 함수들
# ============================================================
@st.cache_resource
def load_embedding_model():
    """임베딩 모델 로드 (캐시됨)"""
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

@st.cache_data
def get_law_articles(law_msn, oc="kangyoon.kim"):
    """법령 조문 가져오기 (캐시됨)"""
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {
        "OC": oc,
        "target": "law",
        "type": "XML",
        "MST": law_msn
    }
    
    response = requests.get(url, params=params)
    response.encoding = 'utf-8'
    root = ET.fromstring(response.content)
    
    articles = []
    
    for article in root.findall('.//조문단위'):
        article_no = article.findtext('조문번호', '')
        article_title = article.findtext('조문제목', '')
        article_content = article.findtext('조문내용', '')
        
        hang_list = []
        for hang in article.findall('.//항'):
            hang_content = hang.findtext('항내용', '')
            if hang_content:
                hang_list.append(hang_content)
        
        full_text = f"제{article_no}조"
        if article_title:
            full_text += f"({article_title})"
        full_text += "\n"
        if article_content:
            full_text += article_content + "\n"
        if hang_list:
            full_text += "\n".join(hang_list)
        
        if article_content or hang_list:
            articles.append({
                "article_no": article_no,
                "title": article_title,
                "full_text": full_text.strip()
            })
    
    return articles

@st.cache_resource
def build_vector_db(_embedding_model, articles):
    """벡터 DB 구축 (캐시됨)"""
    chroma_client = chromadb.Client()
    
    try:
        chroma_client.delete_collection("osh_law")
    except:
        pass
    
    collection = chroma_client.create_collection(name="osh_law")
    
    for idx, article in enumerate(articles):
        text = article['full_text']
        embedding = _embedding_model.encode(text).tolist()
        
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"article_no": article['article_no'], "title": article['title'] or ""}],
            ids=[f"article_{idx}"]
        )
    
    return collection

def search_law(query, collection, embedding_model, n_results=3):
    """관련 조문 검색"""
    query_embedding = embedding_model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

def ask_chatbot(question, collection, embedding_model, api_key):
    """챗봇 질문-답변"""
    # 관련 조문 검색
    search_results = search_law(question, collection, embedding_model)
    context = "\n\n---\n\n".join(search_results['documents'][0])
    
    # Claude API 호출
    client = Anthropic(api_key=api_key)
    
    prompt = f"""당신은 산업안전보건법 전문 상담사입니다.
아래 제공된 법령 조문을 참고하여 질문에 답변해주세요.

## 참고 법령 조문:
{context}

## 질문:
{question}

## 답변 지침:
1. 반드시 위 조문 내용을 근거로 답변하세요
2. 관련 조문 번호를 명시하세요 (예: 제29조에 따르면...)
3. 조문에 없는 내용은 "해당 내용은 제공된 조문에서 찾지 못했습니다"라고 답하세요
4. 쉽고 친절하게 설명하세요
5. 마지막에 면책조항을 추가하세요: "※ 본 답변은 참고용이며, 정확한 법률 해석은 전문가와 상담하시기 바랍니다."

## 답변:"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text, search_results['documents'][0]

# ============================================================
# 메인 UI
# ============================================================
st.title("🏭 산업안전보건법 AI 상담사")
st.markdown("산업안전보건법에 대해 무엇이든 질문하세요!")

# 사이드바 - API 키 입력
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Claude API Key", type="password", help="Anthropic API 키를 입력하세요")
    
    st.markdown("---")
    st.markdown("### 📚 지원 법령")
    st.markdown("- 산업안전보건법")
    st.markdown("- *(추후 추가 예정)*")
    
    st.markdown("---")
    st.markdown("### ℹ️ 안내")
    st.markdown("""
    이 챗봇은 산업안전보건법 조문을 
    기반으로 답변합니다.
    
    **면책조항**: 본 서비스는 참고용이며, 
    법률적 효력이 없습니다.
    """)

# 시스템 초기화
if api_key:
    with st.spinner("🔄 시스템 초기화 중... (처음 한 번만 시간이 걸립니다)"):
        # 모델 로드
        embedding_model = load_embedding_model()
        
        # 법령 데이터 로드
        articles = get_law_articles("276853")
        
        # 벡터 DB 구축
        collection = build_vector_db(embedding_model, articles)
        
        st.session_state.initialized = True
    
    st.success(f"✅ 준비 완료! 산업안전보건법 {len(articles)}개 조문 로드됨")
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("🔍 관련 조문 검색 중..."):
                try:
                    answer, references = ask_chatbot(
                        prompt, collection, embedding_model, api_key
                    )
                    st.markdown(answer)
                    
                    # 참고 조문 표시
                    with st.expander("📜 참고한 법령 조문 보기"):
                        for i, ref in enumerate(references, 1):
                            st.markdown(f"**조문 {i}**")
                            st.text(ref[:500] + "..." if len(ref) > 500 else ref)
                            st.markdown("---")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")

else:
    st.warning("👈 왼쪽 사이드바에서 Claude API Key를 입력해주세요.")
    
    st.markdown("---")
    st.markdown("### 🚀 시작하기")
    st.markdown("""
    1. [Anthropic Console](https://console.anthropic.com/)에서 API 키 발급
    2. 왼쪽 사이드바에 API 키 입력
    3. 질문 시작!
    """)
    
    st.markdown("---")
    st.markdown("### 💡 질문 예시")
    st.markdown("""
    - 사업주의 안전보건교육 의무는 무엇인가요?
    - 안전관리자 선임 기준이 어떻게 되나요?
    - 도급인의 안전보건 조치 의무는?
    - 산업재해 발생 시 보고 의무는?
    """)

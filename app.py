"""
🏭 안전환경 법규 AI 상담사
Streamlit 웹 앱 (진짜 최종 완전체 버전)

포함 법령: 25개
포함 고시: 8개
총: 33개 법규
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
    page_title="안전환경 법규 AI 상담사",
    page_icon="🏭",
    layout="centered"
)

# ============================================================
# 🔐 API 키 설정
# ============================================================
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]

# ============================================================
# 세션 상태 초기화
# ============================================================
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ============================================================
# 📚 법령/고시 목록 정의
# ============================================================

# 법령 (target: law)
LAWS = [
    # ========== 산업안전보건 ==========
    ("276853", "산업안전보건법"),
    ("277411", "산업안전보건법 시행령"),
    ("271485", "산업안전보건법 시행규칙"),
    ("277059", "산업안전보건기준에 관한 규칙"),
    
    # ========== 화학물질관리법 (화관법) ==========
    ("276815", "화학물질관리법"),
    ("280507", "화학물질관리법 시행령"),
    ("279031", "화학물질관리법 시행규칙"),
    
    # ========== 화평법 ==========
    ("279805", "화학물질의 등록 및 평가 등에 관한 법률"),
    ("280633", "화학물질의 등록 및 평가 등에 관한 법률 시행령"),
    ("282061", "화학물질의 등록 및 평가 등에 관한 법률 시행규칙"),
    
    # ========== 위험물안전관리법 ==========
    ("259933", "위험물안전관리법"),
    ("273077", "위험물안전관리법 시행령"),
    ("262765", "위험물안전관리법 시행규칙"),
    
    # ========== 고압가스 안전관리법 ==========
    ("276461", "고압가스 안전관리법"),
    ("278293", "고압가스 안전관리법 시행령"),
    ("278693", "고압가스 안전관리법 시행규칙"),
    
    # ========== 대기환경보전법 ==========
    ("279785", "대기환경보전법"),
    ("280555", "대기환경보전법 시행령"),
    ("280747", "대기환경보전법 시행규칙"),
    
    # ========== 물환경보전법 ==========
    ("276739", "물환경보전법"),
    ("281847", "물환경보전법 시행령"),
    ("282047", "물환경보전법 시행규칙"),
    
    # ========== 폐기물관리법 ==========
    ("279797", "폐기물관리법"),
    ("282339", "폐기물관리법 시행령"),
    ("282261", "폐기물관리법 시행규칙"),
]

# 행정규칙/고시 (target: admrul)
ADMIN_RULES = [
    # ========== 산업안전보건 고시 ==========
    ("2100000251014", "사업장 위험성평가에 관한 지침"),
    ("2100000262720", "화학물질의 분류·표시 및 물질안전보건자료에 관한 기준"),
    ("2100000186058", "화학물질 및 물리적 인자의 노출기준"),
    ("2100000186111", "작업환경측정 및 정도관리 등에 관한 고시"),
    ("2100000186112", "사무실 공기관리 지침"),
    
    # ========== 위험물 고시 ==========
    ("2100000249286", "위험물안전관리에 관한 세부기준"),
    
    # ========== 고압가스 고시 ==========
    ("2100000211965", "고압가스안전관리기준통합고시"),
    
    # ========== 화학물질 고시 ==========
    ("2100000262588", "유해화학물질별 구체적인 취급기준에 관한 규정"),
]

# ============================================================
# 법령 데이터 로드 함수들
# ============================================================
@st.cache_resource
def load_embedding_model():
    """임베딩 모델 로드"""
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

@st.cache_data
def get_law_data(law_msn, law_name, target="law", oc="kangyoon.kim"):
    """법령/고시 조문 + 별표 가져오기"""
    
    if target == "law":
        url = "http://www.law.go.kr/DRF/lawService.do"
    else:
        url = "http://www.law.go.kr/DRF/admRulService.do"
    
    params = {
        "OC": oc,
        "target": target,
        "type": "XML",
        "MST": law_msn
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.encoding = 'utf-8'
        root = ET.fromstring(response.content)
    except Exception as e:
        return []
    
    all_data = []
    
    # 1. 조문 가져오기
    for article in root.findall('.//조문단위'):
        article_no = article.findtext('조문번호', '')
        article_title = article.findtext('조문제목', '')
        article_content = article.findtext('조문내용', '')
        
        hang_list = []
        for hang in article.findall('.//항'):
            hang_content = hang.findtext('항내용', '')
            if hang_content:
                hang_list.append(hang_content)
        
        full_text = f"[{law_name}] 제{article_no}조"
        if article_title:
            full_text += f"({article_title})"
        full_text += "\n"
        if article_content:
            full_text += article_content + "\n"
        if hang_list:
            full_text += "\n".join(hang_list)
        
        if article_content or hang_list:
            all_data.append({
                "type": "조문",
                "law_name": law_name,
                "number": f"제{article_no}조",
                "title": article_title or "",
                "full_text": full_text.strip()
            })
    
    # 2. 별표 가져오기
    for bt in root.findall('.//별표단위'):
        bt_no = bt.findtext('별표번호', '')
        bt_title = bt.findtext('별표제목', '')
        bt_content = bt.findtext('별표내용', '')
        
        if bt_content and len(bt_content) > 50:
            full_text = f"[{law_name}] [별표 {bt_no}] {bt_title}\n\n{bt_content}"
            
            all_data.append({
                "type": "별표",
                "law_name": law_name,
                "number": f"별표 {bt_no}",
                "title": bt_title or "",
                "full_text": full_text.strip()
            })
    
    return all_data

@st.cache_data
def get_all_data():
    """모든 법령/고시 데이터 통합"""
    all_data = []
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    total = len(LAWS) + len(ADMIN_RULES)
    
    # 법령 로드
    for i, (msn, name) in enumerate(LAWS):
        progress_text.text(f"📥 {name} 로드 중...")
        progress_bar.progress((i + 1) / total)
        data = get_law_data(msn, name, target="law")
        all_data.extend(data)
    
    # 행정규칙/고시 로드
    for i, (msn, name) in enumerate(ADMIN_RULES):
        progress_text.text(f"📥 {name} 로드 중...")
        progress_bar.progress((len(LAWS) + i + 1) / total)
        data = get_law_data(msn, name, target="admrul")
        all_data.extend(data)
    
    progress_text.empty()
    progress_bar.empty()
    return all_data

@st.cache_resource
def build_vector_db(_embedding_model, all_data):
    """벡터 DB 구축"""
    chroma_client = chromadb.Client()
    
    try:
        chroma_client.delete_collection("osh_law")
    except:
        pass
    
    collection = chroma_client.create_collection(name="osh_law")
    
    for idx, item in enumerate(all_data):
        text = item['full_text']
        
        if len(text) > 2000:
            text = text[:2000]
        
        embedding = _embedding_model.encode(text).tolist()
        
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "type": item['type'],
                "law_name": item['law_name'],
                "number": item['number'],
                "title": item['title']
            }],
            ids=[f"item_{idx}"]
        )
    
    return collection

def search_law(query, collection, embedding_model, n_results=5):
    """관련 조문/별표 검색"""
    query_embedding = embedding_model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

def ask_chatbot(question, collection, embedding_model):
    """챗봇 질문-답변"""
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
2. 출처를 명확히 밝히세요 (예: "산업안전보건법 제29조에 따르면...", "위험물안전관리에 관한 세부기준 제5조에 따르면...")
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
    
    with st.expander("🔧 산업안전보건 (4개 법령)", expanded=False):
        st.markdown("""
        - 산업안전보건법
        - 시행령 / 시행규칙
        - 산업안전보건기준에 관한 규칙
        """)
    
    with st.expander("🧪 화학물질 (6개 법령)", expanded=False):
        st.markdown("""
        - 화학물질관리법 (화관법)
        - 화평법
        - 각 시행령/시행규칙
        """)
    
    with st.expander("🔥 위험물/고압가스 (6개 법령)", expanded=False):
        st.markdown("""
        - 위험물안전관리법
        - 고압가스 안전관리법
        - 각 시행령/시행규칙
        """)
    
    with st.expander("🌿 환경 (9개 법령)", expanded=False):
        st.markdown("""
        - 대기환경보전법
        - 물환경보전법
        - 폐기물관리법
        - 각 시행령/시행규칙
        """)
    
    with st.expander("📋 고시/지침 (8개)", expanded=False):
        st.markdown("""
        **산안법 관련**
        - 위험성평가 지침
        - MSDS 기준
        - 노출기준
        - 작업환경측정 고시
        - 사무실 공기관리 지침
        
        **기타**
        - 위험물안전관리 세부기준
        - 고압가스안전관리기준통합고시
        - 유해화학물질 취급기준
        """)
    
    st.markdown("---")
    
    st.header("💡 질문 예시")
    st.markdown("""
    **산업안전**
    - 안전관리자 선임 기준은?
    - 위험성평가 절차는?
    
    **화학물질**
    - MSDS 작성 기준은?
    - 유해화학물질 취급기준은?
    
    **위험물**
    - 위험물 저장소 기준은?
    - 위험물 안전거리는?
    
    **환경**
    - 대기배출허용기준은?
    - 폐기물 처리 기준은?
    """)
    
    st.markdown("---")
    
    st.markdown("""
    **⚠️ 면책조항**  
    본 서비스는 참고용이며, 
    법률적 효력이 없습니다.
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ by 힐스")

# 시스템 초기화
with st.spinner("🔄 시스템 준비 중..."):
    embedding_model = load_embedding_model()
    all_data = get_all_data()
    collection = build_vector_db(embedding_model, all_data)

# 통계 표시
article_count = len([d for d in all_data if d['type'] == '조문'])
table_count = len([d for d in all_data if d['type'] == '별표'])
law_count = len(LAWS) + len(ADMIN_RULES)
st.success(f"✅ 준비 완료! ({law_count}개 법규 | 조문 {article_count:,}개 + 별표 {table_count}개)")

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 관련 법규 검색 중..."):
            try:
                answer, search_results = ask_chatbot(prompt, collection, embedding_model)
                st.markdown(answer)
                
                with st.expander("📜 참고한 법규 자료 보기"):
                    for i, (doc, meta) in enumerate(zip(
                        search_results['documents'][0], 
                        search_results['metadatas'][0]
                    ), 1):
                        badge = "📋" if meta['type'] == '조문' else "📊"
                        st.markdown(f"**{badge} {meta['law_name']} {meta['number']}** - {meta['title']}")
                        st.text(doc[:600] + "..." if len(doc) > 600 else doc)
                        st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

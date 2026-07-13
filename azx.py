import gc
import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------------
# 페이지 설정 및 CSS (모바일 최적화 및 UI 다듬기)
# ------------------------------------------------------------------
st.set_page_config(page_title="헬스잇 - AI 알약 탐지", page_icon="💊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        max-width: 100%;
        padding: 0.35rem 0.35rem 2rem;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }
    [data-testid="stFileUploader"] > section {
        padding: 0.6rem;
        border-radius: 0.8rem;
    }
    [data-testid="stBaseButton"] button {
        min-height: 44px;
        border-radius: 0.8rem;
    }
    /* 스플래시 및 로딩 센터링 스타일 */
    .splash-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 100px 20px;
    }
    .splash-logo {
        font-size: 4rem;
        margin-bottom: 20px;
    }
    @media (max-width: 768px) {
        .stApp { padding: 0.2rem 0.2rem 1.2rem; }
        .block-container {
            padding-top: 0.6rem !important;
            padding-left: 0.25rem !important;
            padding-right: 0.25rem !important;
        }
        h1 { font-size: 1.65rem !important; line-height: 1.2 !important; }
        h2 { font-size: 1.25rem !important; }
        .stImage img { border-radius: 0.75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 세션 상태(Session State) 초기화
# ------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = "1_SPLASH"  # 시작 단계
if "raw_image" not in st.session_state:
    st.session_state.raw_image = None
if "detections" not in st.session_state:
    st.session_state.detections = []
if "output_img_rgb" not in st.session_state:
    st.session_state.output_img_rgb = None

# ------------------------------------------------------------------
# 설정값 (사이드바)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    model_path = st.text_input("모델 가중치 경로 (.pt)", value="best.pt")
    drug_info_path = st.text_input(
        "의약품 추가정보 CSV 경로 (선택)",
        value="drug_info.csv",
        help="컬럼 필수: name, company, category, effect, side_effect",
    )
    conf_threshold = st.slider("신뢰도(confidence) 임계값", 0.05, 0.95, 0.25, 0.05)
    
    st.markdown("---")
    if st.button("🔄 처음부터 다시 시작 (앱 초기화)"):
        st.session_state.step = "1_SPLASH"
        st.session_state.raw_image = None
        st.session_state.detections = []
        st.session_state.output_img_rgb = None
        st.rerun()

PAD_SIZE = 1280
OUT_SIZE = 640

# ------------------------------------------------------------------
# 데이터 및 모델 로드 함수
# ------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)

@st.cache_data
def load_drug_info(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # 인덱스를 name으로 설정하고 딕셔너리로 변환
    return df.set_index("name").to_dict(orient="index")

# 모델 로드 프로세스
model = None
model_load_error = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        model_load_error = str(e)
else:
    model_load_error = f"파일을 찾을 수 없습니다: {model_path}"

drug_info = load_drug_info(drug_info_path)


# ==================================================================
# 1단계: 스플래시 창 (Splash Screen)
# ==================================================================
if st.session_state.step == "1_SPLASH":
    st.markdown(
        """
        <div class="splash-container">
            <div class="splash-logo">💊</div>
            <h1>헬스잇 (Health Eat)</h1>
            <p style="color: gray; font-size: 1.1rem;">AI 기반 스마트 알약 탐지 솔루션</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # 모델 로드 에러가 있다면 미리 알려줌
    if model_load_error:
        st.error(f"🚨 시스템 모델 로드 실패: {model_load_error}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        if st.button("🚀 서비스 시작하기", use_container_width=True, type="primary"):
            st.session_state.step = "2_INPUT"
            st.rerun()


# ==================================================================
# 2단계: 사진 입력 창 (카메라 촬영 / 파일 업로드)
# ==================================================================
elif st.session_state.step == "2_INPUT":
    st.title("💊 헬스잇 - 사진 입력")
    st.caption("AI 알고리즘이 알약을 인식할 수 있도록 사진을 제공해 주세요.")
    
    if drug_info is None:
        st.info(f"ℹ️ `{drug_info_path}` 파일이 없어 기본 탐지 정보(이름)만 제공됩니다. (제조사/계열/부작용 제외)")

    st.markdown("---")
    
    # 모바일 환경을 고려하여 카메라와 업로더를 탭으로 분리하여 깔끔하게 배치
    tab_camera, tab_upload = st.tabs(["📸 카메라로 찍기", "📁 파일 업로드"])
    
    uploaded_file = None
    
    with tab_camera:
        # 모바일 기기에서 호출 시 기본 후면 카메라 등이 연동됩니다.
        camera_file = st.camera_input("알약을 화면 중앙에 맞춰 찍어주세요")
        if camera_file:
            uploaded_file = camera_file

    with tab_upload:
        file_input = st.file_uploader(
            "갤러리나 파일에서 알약 사진을 선택하세요.", type=["jpg", "jpeg", "png"]
        )
        if file_input:
            uploaded_file = file_input

    # 사진이 입력되면 세션 변수에 저장하고 다음 단계(로딩)로 강제 전송
    if uploaded_file is not None:
        uploaded_file.seek(0)
        st.session_state.raw_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.step = "3_LOADING"
        st.rerun()


# ==================================================================
# 3단계: 인식 진행 로딩창 (Spinner 애니메이션)
# ==================================================================
elif st.session_state.step == "3_LOADING":
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 동그라미 스피너와 진행 상태 메시지 출력
    with st.spinner("🔄 알약을 인식하는 중입니다. 잠시만 기다려주세요..."):
        if st.session_state.raw_image is not None and model is not None:
            
            # [백엔드 오리지널 로직 수행]
            orig = cv2.cvtColor(np.array(st.session_state.raw_image), cv2.COLOR_RGB2BGR)
            oh, ow = orig.shape[:2]

            pad_w = max(PAD_SIZE - ow, 0)
            pad_h = max(PAD_SIZE - oh, 0)
            left = pad_w // 2
            top = pad_h // 2

            padded = cv2.copyMakeBorder(
                orig, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
            resized = cv2.resize(padded, (OUT_SIZE, OUT_SIZE))

            # YOLOv8 추론
            results = model.predict(resized, imgsz=OUT_SIZE, conf=conf_threshold, verbose=False)[0]

            scale = PAD_SIZE / OUT_SIZE
            output_img = orig.copy()
            local_detections = []

            for box in results.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = x1 * scale - left
                y1 = y1 * scale - top
                x2 = x2 * scale - left
                y2 = y2 * scale - top

                x1 = max(0, min(x1, ow))
                y1 = max(0, min(y1, oh))
                x2 = max(0, min(x2, ow))
                y2 = max(0, min(y2, oh))

                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue

                drug_name = model.names.get(cls_idx, f"알 수 없는 클래스 ({cls_idx})")
                det_id = len(local_detections) + 1

                # 박스 및 넘버링 그리기
                cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(
                    output_img, str(det_id), (int(x1), max(int(y1) - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
                )

                local_detections.append({
                    "id": det_id,
                    "cls_idx": cls_idx,
                    "drug_name": drug_name,
                    "conf": conf,
                })

            # 데이터 저장 및 메모리 정리
            st.session_state.detections = local_detections
            st.session_state.output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            
            del padded, resized, results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 사용자에게 자연스러운 UX를 주기 위해 살짝 딜레이 후 결과창 이동
            time.sleep(0.5) 
            st.session_state.step = "4_RESULT"
            st.rerun()
        else:
            st.error("오류: 이미지 정보가 없거나 모델이 로드되지 않았습니다.")
            if st.button("처음으로 돌아가기"):
                st.session_state.step = "1_SPLASH"
                st.rerun()


# ==================================================================
# 4단계: 결과 출력 창 (탐지 내역, 상세 리포트, 화살표 토글 부작용)
# ==================================================================
elif st.session_state.step == "4_RESULT":
    st.title("🔍 AI 알약 인식 결과")
    
    st.warning(
        "⚠️ **이 결과는 AI 모델의 예측값이며 100% 정확하지 않을 수 있습니다.**\n\n"
        "실제로 복용 중인 약을 확인하려면 반드시 약사·의사와 상담하거나 의약품안전나라에서 재확인해 주세요."
    )
    
    col_res_left, col_res_right = st.columns([1, 1])
    
    with col_res_left:
        st.subheader("🖼️ 인식된 이미지")
        if st.session_state.output_img_rgb is not None:
            st.image(st.session_state.output_img_rgb, caption="AI 탐지 결과 바운딩 박스", use_container_width=True)
        
        # 재촬영 편리성을 위해 결과창 아래에 바로 배치
        if st.button("🔄 다른 사진 찍기 / 다시 하기", use_container_width=True):
            st.session_state.step = "2_INPUT"
            st.session_state.raw_image = None
            st.session_state.detections = []
            st.session_state.output_img_rgb = None
            st.rerun()

    with col_res_right:
        st.subheader("📋 탐지된 알약 리스트")
        detections = st.session_state.detections
        
        if detections:
            st.success(f"🎉 총 {len(detections)}개의 알약을 안정적으로 탐지했습니다.")
            
            for det in detections:
                info = drug_info.get(det["drug_name"]) if drug_info else None
                
                # 아코디언 타이틀 구성
                title = f"[{det['id']}번 알약] {det['drug_name']} (신뢰도: {det['conf'] * 100:.1f}%)"
                
                with st.expander(title, expanded=True):
                    # 5. 약품명 / 6. 무슨 계열약인지 / 7. 효능 출력
                    st.markdown(f"**💊 약품명:** {det['drug_name']}")
                    
                    if info:
                        st.markdown(f"**🏢 제조회사:** {info.get('company', '-')}")
                        st.markdown(f"**🧪 약품 계열:** {info.get('category', '-')}")
                        st.markdown(f"**🎯 효능·효과:** {info.get('effect', '-')}")
                        
                        # --- 추가 정보창 (부작용 화살표 토글 형식) ---
                        # st.expander 내부의 st.expander는 화살표 구조의 완벽한 닫기/열기 서브토글 기능을 수행합니다.
                        with st.expander("🔻 추가 정보 (부작용 확인하기)", expanded=False):
                            st.markdown(
                                f"<span style='color:#d9534f;'>⚠️ <b>주요 부작용 및 주의사항:</b></span><br>{info.get('side_effect', '등록된 부작용 정보가 없습니다.')}", 
                                unsafe_allow_html=True
                            )
                    else:
                        st.caption(
                            f"ℹ️ CSV에 `{det['drug_name']}` 정보가 없어 제조사, 계열, 효능 및 부작용을 불러올 수 없습니다."
                        )
        else:
            st.warning("탐지된 알약이 없습니다. 신뢰도 임계값을 낮추거나 조명을 조절하여 다시 촬영해 주세요.")
            
    st.markdown("---")
    st.info("💡 식약처 의약품안전나라(nedrug.mfds.go.kr) 데이터를 활용하시면 더욱 안전합니다.")
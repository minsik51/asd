import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------
# 1. 초기 세팅 및 데이터 로드
# --------------------------------------------------------------------------
st.set_page_config(page_title="Health-Eat 알약 인식 서비스", layout="wide")

# 세션 상태 초기화
if "detections" not in st.session_state:
    st.session_state.detections = None
if "output_rgb" not in st.session_state:
    st.session_state.output_rgb = None

# 파일 경로 정의 (실제 업로드된 파일명으로 수정 완료)
MODEL_PATH = "best.pt"
DRUG_INFO_PATH = "drug_full_info.csv"


@st.cache_resource
def load_yolo_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    else:
        st.error(f"⚠️ 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
        return None


model = load_yolo_model()


# --------------------------------------------------------------------------
# 2. 로직 및 데이터 처리 함수
# --------------------------------------------------------------------------
def reset_flow():
    st.session_state.detections = None
    st.session_state.output_rgb = None
    st.rerun()


def process_detection(uploaded_file, conf_threshold):
    if model is None:
        return

    # 이미지 변환
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # YOLO 추론
    results = model.predict(source=img_bgr, conf=conf_threshold, save=False)
    result = results[0]

    detections = []
    output_bgr = img_bgr.copy()

    # 바운딩 박스 그리기 및 결과 저장
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        drug_name = model.names[cls_id]

        detections.append(
            {"id": i + 1, "drug_name": drug_name, "conf": conf, "box": (x1, y1, x2, y2)}
        )

        # 이미지에 박스 및 텍스트 시각화
        cv2.rectangle(output_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"#{i + 1} {drug_name} ({conf * 100:.1f}%)"
        cv2.putText(
            output_bgr,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    # 세션 상태에 저장
    st.session_state.detections = detections
    st.session_state.output_rgb = output_rgb


# --------------------------------------------------------------------------
# 3. 메인 화면 레이아웃 (업로드)
# --------------------------------------------------------------------------
def render_upload_form():
    st.title("💊 Health-Eat 알약 인식 및 부작용 가이드")
    st.markdown(
        "알약 사진을 업로드하면 AI가 인식하여 효능 및 주의사항(부작용)을 안내해 드립니다."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "알약 사진을 업로드 하세요 (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"]
        )
    with col2:
        conf_threshold = st.slider(
            "인식 신뢰도 설정 (너무 낮으면 오검출이 생길 수 있습니다)",
            0.1,
            1.0,
            0.25,
            0.05,
        )

    if uploaded_file is not None:
        if st.button("🔍 알약 분석 시작", use_container_width=True):
            with st.spinner("AI 모델이 알약을 분석하는 중입니다..."):
                process_detection(uploaded_file, conf_threshold)
            st.rerun()


# --------------------------------------------------------------------------
# 4. 결과 출력 화면 (데이터 연동 정밀 매핑 보완판)
# --------------------------------------------------------------------------
def render_result() -> None:
    detections = st.session_state.get("detections")
    output_rgb = st.session_state.get("output_rgb")

    drug_df = None
    if os.path.exists(DRUG_INFO_PATH):
        try:
            # CSV 불러오기 및 결측치 처리
            drug_df = pd.read_csv(DRUG_INFO_PATH).fillna("-")

            # 원본 컬럼 존재 여부 체크 보완
            if "dl_name" in drug_df.columns:
                drug_df["clean_dl_name"] = (
                    drug_df["dl_name"].astype(str).str.replace(r"\s+", "", regex=True)
                )
            else:
                st.error(
                    f"⚠️ CSV 파일에 'dl_name' 컬럼이 존재하지 않습니다. 현재 컬럼: {list(drug_df.columns)}"
                )
                drug_df = None

        except Exception as e:
            st.error(f"⚠️ CSV 데이터를 불러오는 중 오류가 발생했습니다: {e}")

    # 상단: 분석된 이미지 표시
    st.subheader("📷 분석 결과 이미지")
    if output_rgb is not None:
        st.image(output_rgb, use_container_width=True)

    if not detections:
        st.warning("인식된 알약이 없습니다.")
    else:
        st.subheader(f"📋 인식된 알약 정보 ({len(detections)}건)")

        for det in detections:
            drug_name = det["drug_name"]
            conf = det["conf"]

            # YOLO 클래스명(모델 인식 이름)의 공백 제거 후 CSV의 clean_dl_name과 매칭
            matched_row = None
            if drug_df is not None:
                clean_name = str(drug_name).replace(" ", "")
                match = drug_df[drug_df["clean_dl_name"] == clean_name]
                if not match.empty:
                    matched_row = match.iloc[0]
                else:
                    # 완전 일치가 없으면 부분 포함 매칭으로 재시도
                    partial = drug_df[
                        drug_df["clean_dl_name"].str.contains(
                            clean_name, na=False, regex=False
                        )
                    ]
                    if not partial.empty:
                        matched_row = partial.iloc[0]

            with st.container(border=True):
                st.markdown(f"### #{det['id']} {drug_name}  `({conf * 100:.1f}%)`")

                if matched_row is not None:
                    dl_name = matched_row.get("dl_name", "-")
                    dl_company = matched_row.get("제조사", "-")
                    class_code = matched_row.get("효능군코드", "-")
                    class_name = matched_row.get("효능군명", "-")
                    interaction_drug = matched_row.get("병용금기_상대약", "-")
                    drug_idx = matched_row.get("idx", "-")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**💊 약 이름**\n\n{dl_name}")
                        st.markdown(f"**🏭 제조사**\n\n{dl_company}")
                        st.markdown(f"**🔖 식별 인덱스(idx)**\n\n{drug_idx}")
                    with c2:
                        st.markdown(
                            f"**🧪 효능군 분류**\n\n[{class_code}] {class_name}"
                        )
                        if interaction_drug != "-":
                            st.markdown(
                                f"**⚠️ 병용금기 상대약**\n\n:red[{interaction_drug}]"
                            )
                        else:
                            st.markdown("**⚠️ 병용금기 상대약**\n\n해당 없음")
                else:
                    st.info("CSV에서 이 약에 대한 상세 정보를 찾을 수 없습니다.")

    st.divider()
    if st.button("🔄 다시 분석하기", use_container_width=True):
        reset_flow()


# --------------------------------------------------------------------------
# 5. 메인 실행 흐름
# --------------------------------------------------------------------------
def main():
    if st.session_state.get("detections") is not None:
        render_result()
    else:
        render_upload_form()


if __name__ == "__main__":
    main()

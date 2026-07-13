from ultralytics import YOLO

# 1. 모델 가중치 파일 로드
model = YOLO("best.pt")

print("=" * 50)
print("🚀 YOLOv8 모델 내부 정보 확인")
print("=" * 50)

# 2. 모델이 맞출 수 있는 클래스(알약) 총 개수와 이름 출력
print(f"📊 학습된 총 클래스 개수: {len(model.names)}개")
print("\n📝 클래스 목록:")
for idx, name in model.names.items():
    print(f"  [{idx}] {name}")

print("-" * 50)

# 3. 모델의 상세 구조(레이어) 및 메타데이터 정보 요약 보기
print("🧠 모델 구조 요약:")
model.info()

print("=" * 50)

# backend_face_id/app/core/config.py
from pathlib import Path

# Xác định thư mục gốc của dự án backend (backend_face_id)
# Path(__file__) là đường dẫn đến file config.py hiện tại
# .resolve() để có đường dẫn tuyệt đối
# .parent để đi lên thư mục cha (app/core/ -> app/ -> backend_face_id/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- Cấu hình cho Mô hình Machine Learning ---
# Đường dẫn đến file model đã huấn luyện (trong thư mục models_ml)
MODEL_NAME = "siamese_branch_model_best.h5"  # Tên file model của bạn
MODEL_PATH = BASE_DIR / "models_ml" / MODEL_NAME

# Ngưỡng tối ưu bạn đã tìm được từ quá trình đánh giá
# THAY THẾ giá trị này bằng ngưỡng thực tế của bạn
# OPTIMAL_THRESHOLD = 0.8  # Ví dụ, bạn cần cập nhật giá trị này
OPTIMAL_THRESHOLD = 1.2  # Ví dụ, bạn cần cập nhật giá trị này

# Kích thước input mà mô hình Siamese branch mong đợi
MODEL_INPUT_SHAPE = (100, 100, 3)  # (Cao, Rộng, Kênh)

# --- Cấu hình cho việc lưu trữ Embedding (cho demo) ---
# Đường dẫn đến file lưu trữ embeddings (ví dụ: file JSON)
EMBEDDINGS_STORE_DIR = BASE_DIR / "app" / "database"
EMBEDDINGS_STORE_FILE_NAME = "embeddings_store.json"
# EMBEDDINGS_STORE_FILE_NAME = "embeddings_store_test.json"
EMBEDDINGS_STORE_PATH = EMBEDDINGS_STORE_DIR / EMBEDDINGS_STORE_FILE_NAME

# Ngưỡng tin cậy để quyết định cuối cùng trong việc xác minh
# Nếu confidence_score >= CONFIDENCE_VERIFICATION_THRESHOLD thì is_same_person = True
# CONFIDENCE_VERIFICATION_THRESHOLD = 0.5
CONFIDENCE_VERIFICATION_THRESHOLD = 0.6

# In ra để kiểm tra đường dẫn (chỉ cho mục đích debug khi chạy file này trực tiếp)
if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"MODEL_INPUT_SHAPE: {MODEL_INPUT_SHAPE}")
    print(f"OPTIMAL_THRESHOLD: {OPTIMAL_THRESHOLD}")
    print(f"EMBEDDINGS_STORE_PATH: {EMBEDDINGS_STORE_PATH}")
    print(f"CONFIDENCE_VERIFICATION_THRESHOLD: {CONFIDENCE_VERIFICATION_THRESHOLD}")

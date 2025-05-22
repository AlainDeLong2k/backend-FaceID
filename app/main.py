# backend_face_id/app/main.py
from fastapi import FastAPI, HTTPException, status
from typing import Optional, List
import base64
import io
import numpy as np

# Import từ các module trong project
from app.core.config import (
    OPTIMAL_THRESHOLD,  # Ngưỡng cho từng cặp embedding
    CONFIDENCE_VERIFICATION_THRESHOLD,  # Ngưỡng cho confidence score cuối cùng
)
from app.models.schemas import (
    FaceRegistrationRequest,
    FaceRegistrationResponse,
    FaceVerificationRequest,
    FaceVerificationResponse,
    ErrorResponse,
)
from app.services.face_verification_service import FaceVerificationService
from app.database.db_handler import EmbeddingStore

# --- Khởi tạo các service và store (giữ nguyên) ---
face_service = FaceVerificationService()
embedding_store = EmbeddingStore()

# Khởi tạo ứng dụng FastAPI (giữ nguyên)
app = FastAPI(
    title="Face ID Service API",
    description="API for face registration and verification using Siamese Network.",
    version="0.1.0",
)


# --- Hàm trợ giúp decode_base64_image và euclidean_distance_numpy (giữ nguyên) ---
def decode_base64_image(base64_string: str) -> Optional[bytes]:
    try:
        if "," in base64_string:
            header, data = base64_string.split(",", 1)
        else:
            data = base64_string
        return base64.b64decode(data)
    except Exception as e:
        print(f"Lỗi khi giải mã base64: {e}")
        return None


def euclidean_distance_numpy(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    if embedding1 is None or embedding2 is None:
        print("LỖI: Một trong hai embedding là None, không thể tính khoảng cách.")
        return float("inf")
    return float(np.sqrt(np.sum(np.square(embedding1 - embedding2))))


# --- API Endpoints ---


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Welcome to Face ID Service! API is running."}


# Endpoint /register_face (giữ nguyên hoặc thay đổi/xóa tùy theo quyết định của bạn
# vì bạn nói sẽ tạo embeddings_store.json bằng script riêng)
# Hiện tại tôi sẽ giữ lại nó với logic add_embedding mới của EmbeddingStore
# để bạn có thể dùng nếu muốn thêm embedding thủ công qua API.
@app.post(
    "/register_face",
    response_model=FaceRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Face Operations"],
    summary="Register a face by adding its embedding to the user's list",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Invalid image data or user ID",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Failed to process image or save embedding",
        },
    },
)
async def register_face(request_data: FaceRegistrationRequest):
    # <<< THÊM VÀO ĐÂY: Loại bỏ khoảng trắng thừa cho user_id >>>
    user_id_cleaned = request_data.user_id.strip()
    if not user_id_cleaned:  # Kiểm tra xem user_id có rỗng sau khi strip không
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID cannot be empty or just whitespace.",
        )
    # <<< KẾT THÚC THÊM VÀO >>>

    image_bytes = decode_base64_image(request_data.image_base64)
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 image data."
        )

    embedding = face_service.get_embedding(image_bytes)
    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract embedding from the image.",
        )

    # Sử dụng user_id_cleaned đã được làm sạch
    success = embedding_store.add_embedding(
        user_id_cleaned, embedding
    )  # <<< SỬA Ở ĐÂY >>>
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save embedding to store.",
        )

    current_user_embeddings = embedding_store.get_embeddings_for_user(
        user_id_cleaned
    )  # <<< SỬA Ở ĐÂY >>>
    num_embeddings_for_user = (
        len(current_user_embeddings) if current_user_embeddings else 0
    )

    return FaceRegistrationResponse(
        message=f"Embedding added for user {user_id_cleaned}. User now has {num_embeddings_for_user} registered embedding(s).",  # <<< SỬA Ở ĐÂY >>>
        user_id=user_id_cleaned,  # <<< SỬA Ở ĐÂY >>>
    )


# <<< --- SỬA ĐỔI LOGIC CHO /verify_face BẮT ĐẦU TỪ ĐÂY --- >>>
@app.post(
    "/verify_face",
    response_model=FaceVerificationResponse,
    tags=["Face Operations"],
    summary="Verify a face against a registered user ID using multiple stored embeddings",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Invalid image data or user ID",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Registered user ID not found or no embeddings",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Failed to process image",
        },
    },
)
async def verify_face(request_data: FaceVerificationRequest):
    """
    Xác minh một ảnh mới (image_base64_to_check) với một user_id đã đăng ký.
    So sánh embedding của ảnh mới với TẤT CẢ embedding đã lưu của user_id đó.
    Tính confidence score và đưa ra quyết định.
    """
    user_id_to_verify_cleaned = request_data.user_id_to_verify.strip()
    if not user_id_to_verify_cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID to verify cannot be empty or just whitespace.",
        )

    # 1. Lấy embedding của ảnh mới cần kiểm tra
    new_image_bytes = decode_base64_image(request_data.image_base64_to_check)
    if not new_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 data for the image to check.",
        )

    new_embedding = face_service.get_embedding(new_image_bytes)
    if new_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract embedding from the new image.",
        )

    # 2. Lấy DANH SÁCH embedding của user_id đã đăng ký
    stored_embeddings_list = embedding_store.get_embeddings_for_user(
        request_data.user_id_to_verify
    )

    if not stored_embeddings_list:  # Kiểm tra None hoặc list rỗng
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User ID '{request_data.user_id_to_verify}' not found or no embeddings registered.",
        )

    # 3. So sánh và Đếm số lần "Khớp"
    match_count = 0
    min_distance_found = float("inf")
    total_stored_embeddings = len(stored_embeddings_list)

    for stored_embedding in stored_embeddings_list:
        distance = euclidean_distance_numpy(new_embedding, stored_embedding)
        min_distance_found = min(
            min_distance_found, distance
        )  # Cập nhật khoảng cách nhỏ nhất

        if distance < OPTIMAL_THRESHOLD:  # OPTIMAL_THRESHOLD từ config.py
            match_count += 1

    # 4. Tính Độ Tin Cậy (Confidence Score)
    confidence_score = (
        match_count / total_stored_embeddings if total_stored_embeddings > 0 else 0.0
    )

    # 5. Đưa ra Quyết định Cuối cùng is_same_person
    is_same_person = (
        confidence_score >= CONFIDENCE_VERIFICATION_THRESHOLD
    )  # Từ config.py

    message = f"Verification processed. Confidence: {confidence_score:.2f}."
    if is_same_person:
        message += " Faces match."
    else:
        message += " Faces DO NOT match."

    return FaceVerificationResponse(
        is_same_person=is_same_person,
        confidence_score=confidence_score,
        min_distance_found=(
            min_distance_found if min_distance_found != float("inf") else None
        ),
        threshold_used_for_distance=OPTIMAL_THRESHOLD,
        confidence_threshold_used=CONFIDENCE_VERIFICATION_THRESHOLD,
        message=message,
    )


# <<< --- KẾT THÚC SỬA ĐỔI LOGIC CHO /verify_face --- >>>


@app.get("/registered_users", response_model=List[str], tags=["Admin/Debug"])
async def get_registered_users():
    return embedding_store.get_all_user_ids()


# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# backend_face_id/app/models/schemas.py
from pydantic import BaseModel
from typing import Optional, List


class ImageInput(BaseModel):
    image_base64: str  # Chuỗi base64 của ảnh


class FaceRegistrationRequest(BaseModel):
    user_id: str
    image_base64: str


class FaceRegistrationResponse(BaseModel):
    message: str
    user_id: str
    # file_path: Optional[str] = None # Tùy chọn nếu bạn muốn trả về nơi lưu ảnh/embedding


class FaceVerificationRequest(BaseModel):
    user_id_to_verify: str  # ID của người đã đăng ký để so sánh
    image_base64_to_check: str  # Ảnh mới cần kiểm tra


class FaceVerificationResponse(BaseModel):
    is_same_person: bool
    confidence_score: float  # << THÊM MỚI
    min_distance_found: Optional[float] = (
        None  # << THAY ĐỔI/LÀM RÕ: Khoảng cách nhỏ nhất tìm được
    )
    # Có thể là None nếu không có embedding nào để so sánh
    # distance: float # Có thể bỏ trường distance cũ này nếu min_distance_found và confidence_score là đủ
    threshold_used_for_distance: float  # Ngưỡng dùng để so sánh từng cặp embedding
    confidence_threshold_used: (
        float  # Ngưỡng dùng để quyết định is_same_person từ confidence_score
    )
    message: str


class ErrorResponse(BaseModel):
    detail: str


# (Sau này có thể thêm các schema khác nếu cần)

# backend_face_id/app/database/db_handler.py
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any  # Thêm Any

# Import đường dẫn file lưu trữ từ config
from app.core.config import EMBEDDINGS_STORE_PATH, EMBEDDINGS_STORE_DIR

# (Logger setup - ví dụ)
import logging

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Cấu hình nếu chạy độc lập


class EmbeddingStore:
    def __init__(self, store_path: Path = EMBEDDINGS_STORE_PATH):
        self.store_path = store_path
        EMBEDDINGS_STORE_DIR.mkdir(parents=True, exist_ok=True)
        # self.embeddings_data sẽ là Dict[str, List[List[float]]]
        # mỗi List[float] bên trong là một embedding
        self.embeddings_data: Dict[str, List[List[float]]] = self._load_store()
        logger.info(
            f"EmbeddingStore initialized. Loaded {len(self.embeddings_data)} users from {self.store_path}"
        )

    def _load_store(self) -> Dict[str, List[List[float]]]:
        """Tải dữ liệu embeddings từ file JSON."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Đảm bảo value của mỗi key là một list (có thể chứa các list embedding)
                    for user_id, embeddings in data.items():
                        if not isinstance(embeddings, list):
                            logger.warning(
                                f"Dữ liệu cho user_id '{user_id}' trong store không phải là list, bỏ qua user này."
                            )
                            # Có thể xóa key này hoặc xử lý khác: data[user_id] = []
                            continue  # Bỏ qua user này nếu định dạng không đúng
                        # Kiểm tra sâu hơn (tùy chọn): mỗi item trong list embedding có phải là list of numbers
                        for emb_list in embeddings:
                            if not (
                                isinstance(emb_list, list)
                                and all(isinstance(x, (int, float)) for x in emb_list)
                            ):
                                logger.warning(
                                    f"Một embedding cho user_id '{user_id}' không phải là list of numbers. Cần kiểm tra file store."
                                )
                                # Có thể lọc bỏ embedding không hợp lệ này
                    logger.info(f"Đã tải {len(data)} users từ {self.store_path}")
                    return data
            except json.JSONDecodeError:
                logger.error(
                    f"Lỗi decode JSON từ {self.store_path}. Trả về store rỗng."
                )
                return {}
            except Exception as e:
                logger.exception(
                    f"Lỗi không xác định khi tải embedding store từ {self.store_path}:"
                )
                return {}
        logger.info(f"File store {self.store_path} không tồn tại. Khởi tạo store rỗng.")
        return {}

    def _save_store(self):
        """Lưu dữ liệu embeddings hiện tại vào file JSON."""
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(self.embeddings_data, f, indent=4)
            logger.info(f"Dữ liệu Embeddings đã được lưu vào {self.store_path}")
        except Exception as e:
            logger.exception(f"Lỗi khi lưu embedding store tại {self.store_path}:")

    def add_embedding(self, user_id: str, embedding: np.ndarray) -> bool:
        """
        Thêm một embedding mới vào danh sách embedding của user_id.
        Nếu user_id chưa tồn tại, tạo mới danh sách.
        """
        if not isinstance(embedding, np.ndarray):
            logger.error(
                f"Attempted to save non-numpy array embedding for user {user_id}"
            )
            return False

        embedding_as_list = (
            embedding.tolist()
        )  # Chuyển NumPy array thành list of floats

        if user_id not in self.embeddings_data:
            self.embeddings_data[user_id] = []

        # (Tùy chọn: Kiểm tra trùng lặp embedding trước khi thêm nếu cần)
        # current_embeddings_for_user = [np.array(e) for e in self.embeddings_data[user_id]]
        # is_duplicate = any(np.array_equal(embedding, existing_emb) for existing_emb in current_embeddings_for_user)
        # if is_duplicate:
        #     logger.info(f"Embedding for user {user_id} already exists. Not adding duplicate.")
        #     return True # Coi như thành công vì đã có

        self.embeddings_data[user_id].append(embedding_as_list)
        self._save_store()
        logger.info(
            f"Đã thêm embedding mới cho user_id: {user_id}. Tổng số embeddings: {len(self.embeddings_data[user_id])}"
        )
        return True

    def get_embeddings_for_user(self, user_id: str) -> Optional[List[np.ndarray]]:
        """
        Lấy danh sách các embedding (dưới dạng NumPy array) của một user_id.
        Trả về list các NumPy array hoặc None nếu không tìm thấy user_id hoặc không có embedding.
        """
        list_of_embedding_lists = self.embeddings_data.get(user_id)
        if list_of_embedding_lists:
            try:
                # Chuyển mỗi list con (embedding) lại thành NumPy array
                numpy_embeddings = [
                    np.array(emb_list, dtype=np.float32)
                    for emb_list in list_of_embedding_lists
                ]
                logger.debug(
                    f"Đã tìm thấy {len(numpy_embeddings)} embeddings cho user_id: {user_id}"
                )
                return numpy_embeddings
            except Exception as e:
                logger.exception(
                    f"Lỗi khi chuyển đổi embeddings sang NumPy array cho user {user_id}:"
                )
                return None  # Hoặc một list rỗng

        logger.warning(
            f"Không tìm thấy user_id '{user_id}' hoặc không có embeddings nào trong store."
        )
        return None  # Hoặc trả về list rỗng: []

    def get_all_user_ids(self) -> List[str]:
        """Trả về danh sách tất cả user_id đã đăng ký."""
        return list(self.embeddings_data.keys())


# --- Phần kiểm tra nhanh (chỉ chạy khi thực thi file này trực tiếp) ---
if __name__ == "__main__":
    # Cấu hình logging cơ bản để thấy output của logger khi chạy độc lập
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(module)s - %(funcName)s: %(message)s",
    )

    # (Phần sys.path.insert như cũ nếu cần để import app.core.config)
    import sys

    current_file_path = Path(__file__).resolve()
    project_root_from_db = current_file_path.parent.parent.parent
    if str(project_root_from_db) not in sys.path:
        sys.path.insert(0, str(project_root_from_db))
        logger.info(f"Đã thêm '{project_root_from_db}' vào sys.path cho import")

    from app.core.config import EMBEDDINGS_STORE_PATH as test_store_path

    # Tạo một file store mẫu (GIẢ ĐỊNH nó được tạo bởi script riêng của bạn)
    sample_store_data = {
        "user_A": [
            np.random.rand(400).astype(np.float32).tolist(),
            np.random.rand(400).astype(np.float32).tolist(),
        ],
        "user_B": [np.random.rand(400).astype(np.float32).tolist()],
    }
    if test_store_path.exists():
        test_store_path.unlink()  # Xóa file cũ nếu có để test _load_store từ đầu
    with open(test_store_path, "w") as f_sample:
        json.dump(sample_store_data, f_sample, indent=4)
    logger.info(f"Đã tạo file store mẫu tại: {test_store_path}")

    print("\n--- Chạy kiểm tra EmbeddingStore với List of Embeddings ---")
    store = EmbeddingStore(store_path=test_store_path)  # Sẽ tải file mẫu ở trên

    print(f"Các user ID đã tải: {store.get_all_user_ids()}")

    # Test get_embeddings_for_user
    embeddings_user_A = store.get_embeddings_for_user("user_A")
    if embeddings_user_A:
        print(f"Số lượng embeddings cho user_A: {len(embeddings_user_A)}")
        print(f"Shape của embedding đầu tiên cho user_A: {embeddings_user_A[0].shape}")
        assert len(embeddings_user_A) == 2
        assert embeddings_user_A[0].shape == (400,)
        assert isinstance(embeddings_user_A[0], np.ndarray)
        print("Kiểm tra get_embeddings_for_user cho user_A: Thành công.")
    else:
        print("Lỗi: Không tải được embeddings cho user_A.")

    embeddings_user_C = store.get_embeddings_for_user("user_C")  # User không tồn tại
    if embeddings_user_C is None:  # Hoặc if not embeddings_user_C
        print(
            "Kiểm tra get_embeddings_for_user cho user không tồn tại: Thành công (trả về None)."
        )
    else:
        print(
            "Lỗi: get_embeddings_for_user cho user không tồn tại nhưng không trả về None."
        )

    # Test add_embedding (thêm vào user_A)
    new_embedding_for_A = np.random.rand(400).astype(np.float32)
    store.add_embedding("user_A", new_embedding_for_A)

    embeddings_user_A_updated = store.get_embeddings_for_user("user_A")
    if embeddings_user_A_updated and len(embeddings_user_A_updated) == 3:
        print(
            f"Số lượng embeddings cho user_A sau khi thêm: {len(embeddings_user_A_updated)}"
        )
        # Kiểm tra xem embedding mới có được thêm vào không (so sánh phần tử cuối)
        assert np.array_equal(embeddings_user_A_updated[-1], new_embedding_for_A)
        print("Kiểm tra add_embedding: Thành công.")
    else:
        print(
            f"Lỗi: add_embedding không thành công hoặc số lượng không đúng (dự kiến 3, có {len(embeddings_user_A_updated) if embeddings_user_A_updated else 'None'})."
        )

    # Test add_embedding cho user mới
    new_embedding_for_D = np.random.rand(400).astype(np.float32)
    store.add_embedding("user_D", new_embedding_for_D)
    embeddings_user_D = store.get_embeddings_for_user("user_D")
    if embeddings_user_D and len(embeddings_user_D) == 1:
        print(f"Số lượng embeddings cho user_D (mới): {len(embeddings_user_D)}")
        assert np.array_equal(embeddings_user_D[0], new_embedding_for_D)
        print("Kiểm tra add_embedding cho user mới: Thành công.")
    else:
        print("Lỗi: add_embedding cho user mới không thành công.")

    print("\n--- Kết thúc kiểm tra EmbeddingStore ---")

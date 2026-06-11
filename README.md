# 🛡️ Ứng dụng Phát hiện Giao dịch Gian lận & Thẩm định Rủi ro

Hệ thống ứng dụng Web được chuyển đổi tự động từ Notebook nghiên cứu mô hình phát hiện rủi ro phân loại sang nền tảng giao diện tương tác trực quan Streamlit. 

Ứng dụng giúp phòng ban chức năng và bộ phận quản trị rủi ro vận hành kiểm thử mô hình học máy **Random Forest Classifier** mà không cần can thiệp vào các khối mã lệnh lập trình.

## ✨ Tính năng chính của hệ thống
- **Quản lý cấu hình linh hoạt (Sidebar):** Tùy chỉnh trực tiếp cấu trúc phân tách cây quyết định, số lượng cây huấn luyện (`n_estimators`), độ sâu cây (`max_depth`), và tỷ lệ phân tách dữ liệu kiểm thử.
- **Phân tích khám phá (Tab 1 & Tab 2):** Khám phá nhanh thông số thống kê mô tả, phân phối tần suất của các thuộc tính biến chỉ báo liên tục từ $X_1$ đến $X_{14}$ tách biệt theo nhãn.
- **Đánh giá hiệu năng chuyên sâu (Tab 3):** Tái hiện đầy đủ ma trận nhầm lẫn trực quan (Confusion Matrix), các thang đo cốt lõi (Accuracy, Precision, Recall, F1-Score) cùng bảng xếp hạng mức độ đóng góp của tính năng (Feature Importance).
- **Thẩm định thực tế (Tab 4):** Hỗ trợ đồng thời 2 luồng công việc: Thẩm định chấm điểm trực tiếp cho một giao dịch đơn lẻ hoặc Tải tập tin quét rủi ro hàng loạt (Batch Processing) và kết xuất tệp báo cáo định dạng Excel/CSV.

## 📁 Cấu trúc dữ liệu đầu vào bắt buộc
Tệp tin tải lên hệ thống (Định dạng hỗ trợ: `.csv`, `.xlsx`) cần tuân thủ cấu trúc phân bổ cột sau:
- **Biến mục tiêu phân loại:** Cột mang tên `default` chứa giá trị nhị phân (`0`: Giao dịch an toàn / `1`: Giao dịch gian lận/rủi ro).
- **Các biến chỉ báo đặc trưng:** Bao gồm 14 cột định lượng số liên tục được đặt tên chính xác từ `X_1`, `X_2`, `X_3`, ..., cho đến `X_14`.

## 🛠️ Hướng dẫn Cài đặt & Chạy ứng dụng

### Bước 1: Khởi tạo và thiết lập môi trường
Đảm bảo máy tính của bạn đã được cài đặt Python phiên bản từ 3.9 đến 3.12. Mở Terminal/Command Prompt tại thư mục chứa mã nguồn và thực thi lệnh cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt

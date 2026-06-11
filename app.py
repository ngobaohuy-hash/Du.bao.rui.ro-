import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import io

# ==============================================================================
# 1) CẤU HÌNH TRANG STREAMLIT ĐẦU TIÊN
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Rủi ro & Gian lận",
    page_icon="🛡️"
)

# ==============================================================================
# 2) IMPORT & CÁC HÀM CACHE DÙNG CHUNG
# ==============================================================================
@st.cache_data(show_spinner="Đang đọc và xử lý dữ liệu...")
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes để tối ưu bộ nhớ đệm cache_data.
    Hỗ trợ cả định dạng CSV và Excel dựa trên đuôi file.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {str(e)}")
        return None

# ==============================================================================
# 3) SIDEBAR (THÀNH PHẦN 1) — VÙNG CẤU HÌNH BỀN VỮNG
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu mẫu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện mẫu", 
        type=["csv", "xlsx"],
        help="Chọn tệp CSV hoặc Excel chứa các biến chỉ báo X và cột nhãn 'default'"
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán mặc định: Random Forest Classifier")
    
    # Trích xuất cấu hình siêu tham số dựa theo kiến trúc cây quyết định của bộ lọc
    n_estimators = st.slider(
        "Số lượng cây quyết định (n_estimators)",
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây quyết định tối đa trong rừng phòng hộ phân loại."
    )
    
    criterion = st.selectbox(
        "Tiêu chí đo lường chất lượng phân tách (criterion)",
        options=["gini", "entropy", "log_loss"],
        index=0,
        help="Hàm đo lường chất lượng phân tách các nút của cây."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa của cây (max_depth)",
        min_value=2, max_value=30, value=15, step=1,
        help="Độ sâu giới hạn của các nhánh cây để kiểm soát hiện tượng quá khớp (overfitting)."
    )
    
    random_state = st.number_input(
        "Hạt giống ngẫu nhiên (random_state)",
        min_value=0, max_value=9999, value=42, step=1,
        help="Đảm bảo tính nhất quán và khả năng tái lập kết quả huấn luyện qua các phiên."
    )
    
    with st.expander("Tham số nâng cao (Advanced Options)"):
        min_samples_split = st.slider("Mẫu tối thiểu để tách nút", 2, 10, 2)
        min_samples_leaf = st.slider("Mẫu tối thiểu tại lá", 1, 10, 1)
        test_size = st.slider("Tỷ lệ chia tập kiểm định (Validation size)", 0.1, 0.5, 0.3, 0.05)

    st.divider()
    # Nút bấm hành động kích hoạt luồng xử lý duy nhất
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==============================================================================
# 4) HEADER (THÀNH PHẦN 2) — VÙNG ĐỊNH HƯỚNG TRẠNG THÁI
# ==============================================================================
st.title("🛡️ Hệ thống Phân tích & Phát hiện Giao dịch Gian lận")
st.caption(
    "Ứng dụng hỗ trợ phòng quản trị rủi ro thẩm định và phân loại các giao dịch bất thường dựa trên mô hình học máy "
    "Random Forest. Đầu vào yêu cầu tập thông tin biến cấu trúc từ X_1 đến X_14 cùng biến mục tiêu 'default'."
)

# Kiểm tra trạng thái dữ liệu đầu vào
if uploaded_file is None:
    st.info("💡 Vui lòng tải file dữ liệu (.csv hoặc .xlsx) tại Sidebar bên trái để bắt đầu phân tích nâng cao.")
    st.stop()
else:
    # Đọc dữ liệu qua cache
    file_bytes = uploaded_file.getvalue()
    df_raw = load_data(file_bytes, uploaded_file.name)
    
    if df_raw is None:
        st.error("Tệp dữ liệu không hợp lệ. Vui lòng kiểm tra lại cấu trúc.")
        st.stop()
        
    st.caption(f"📁 **Đang dùng tệp dữ liệu:** `{uploaded_file.name}`")

st.divider()

# XÁC ĐỊNH SCHEMA BIẾN DỰA TRÊN NOTEBOOK VÀ DỮ LIỆU ĐÍNH KÈM
expected_features = [f"X_{i}" for i in range(1, 15)]
target_col = 'default'

# Kiểm tra tính toàn vẹn của cấu trúc bảng
missing_cols = [col for col in expected_features + ([target_col] if target_col in df_raw.columns else []) if col not in df_raw.columns]
if missing_cols:
    st.error(f"❌ Dữ liệu tải lên thiếu các cột bắt buộc sau: {missing_cols}")
    st.stop()

# ==============================================================================
# 5) KHỐI HUẤN LUYỆN VÀ LƯU TRỮ TRẠNG THÁI (SESSION STATE)
# ==============================================================================
if train_clicked:
    with st.spinner("Mô hình đang học tập dữ liệu và tính toán phân phối xác suất rủi ro..."):
        try:
            # Tách đặc trưng và nhãn mục tiêu
            X = df_raw[expected_features]
            y = df_raw[target_col]
            
            # Phân tách tập dữ liệu thành tập Train/Test theo tham số người dùng đặt
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Khởi tạo thuật toán phân loại theo tham số tùy chỉnh từ Sidebar
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1
            )
            
            # Huấn luyện mô hình
            clf.fit(X_train, y_train)
            
            # Dự báo trên tập kiểm định để đánh giá chỉ tiêu
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
            
            # Lưu trữ vào Session State
            st.session_state['trained_model'] = clf
            st.session_state['features_list'] = expected_features
            st.session_state['evaluation_metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist() if y_proba is not None else None
            }
            st.success("🎉 Huấn luyện mô hình thành công! Chuyển sang các Tab bên dưới để xem kết quả chi tiết.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình huấn luyện: {str(e)}")

# ==============================================================================
# 6) KHỐI NỘI DUNG CHÍNH CHIA TAB
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa biến chỉ báo", 
    "🎯 Kết quả huấn luyện & Kiểm định", 
    "🔮 Ứng dụng dự báo rủi ro"
])

# ------------------------------------------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Phân tích cấu trúc file dữ liệu")
    
    # Tính kích thước dung lượng ước tính
    file_size_mb = len(file_bytes) / (1024 * 1024)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Số lượng quan sát (Dòng)", f"{df_raw.shape[0]:,}")
    with col_m2:
        st.metric("Số lượng biến số (Cột)", f"{df_raw.shape[1]:,}")
    with col_m3:
        st.metric("Dung lượng tệp tin", f"{file_size_mb:.2f} MB")
        
    st.write("### Trích xuất 5 bản ghi dữ liệu đầu tiên (Raw Data)")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.write("### Thống kê mô tả đặc trưng hình học mô hình (X & y)")
    # Chỉ hiển thị mô tả cho các biến tham gia trực tiếp vào mô hình
    st.dataframe(df_raw[expected_features + [target_col]].describe().T, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA BIẾN CHỈ BÁO
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Phân tích phân phối biểu đồ tĩnh và động")
    
    # Biến mục tiêu phân loại: Đặt lên ưu tiên đầu tiên
    st.write("#### 1. Tỷ lệ phân bổ biến mục tiêu rủi ro (`default`)")
    target_counts = df_raw[target_col].value_counts().reset_index()
    target_counts.columns = [target_col, 'Số lượng']
    target_counts[target_col] = target_counts[target_col].astype(str).map({'0': '0 (An toàn/Bình thường)', '1': '1 (Rủi ro/Gian lận)'})
    
    fig_target = px.bar(
        target_counts, x=target_col, y='Số lượng',
        color=target_col, text_auto='.s',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_target.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.write("#### 2. Biểu đồ phân phối các biến chỉ báo đầu vào (Mặc định 4 biến đầu tiên)")
    # Thêm multiselect nếu danh sách biến quá rộng
    selected_features = st.multiselect(
        "Chọn các biến đặc trưng muốn trực quan hóa phân phối dữ liệu:",
        options=expected_features,
        default=expected_features[:4],
        max_selections=8
    )
    
    if selected_features:
        # Bố trí lưới biểu đồ cân đối dạng 2 cột
        cols = st.columns(2)
        for idx, feat in enumerate(selected_features):
            col_target = cols[idx % 2]
            with col_target:
                # Kiểm tra phân phối dữ liệu bằng Box plot và Histogram kết hợp
                fig_feat = px.histogram(
                    df_raw, x=feat, color=df_raw[target_col].astype(str),
                    marginal="box", barmode="overlay",
                    labels={'color': 'Nhãn default'},
                    title=f"Phân phối tần suất đặc trưng {feat} theo nhãn trạng thái",
                    color_discrete_sequence=["#1f77b4", "#ff7f0e"]
                )
                fig_feat.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Vui lòng chọn ít nhất một biến chỉ báo đặc trưng để vẽ biểu đồ.")

# ------------------------------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Chỉ số hiệu năng mô hình phân loại nhị phân")
    
    # Kiểm tra điều phối trạng thái huấn luyện
    if 'trained_model' not in st.session_state:
        st.info("⚠️ Mô hình hiện tại chưa được kích hoạt huấn luyện trên cấu hình này. "
                "Vui lòng nhấn nút '🚀 Huấn luyện mô hình' ở Sidebar bên trái để xem kết quả kiểm định.")
    else:
        metrics = st.session_state['evaluation_metrics']
        
        # Hiển thị các thông số chỉ tiêu vô hướng chính dạng KPI Metric Card
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Độ chính xác tổng thể (Accuracy)", f"{metrics['accuracy']:.2%}")
        c2.metric("Độ chính xác mô hình (Precision)", f"{metrics['precision']:.2%}", help="Khả năng dự báo chính xác trong số các giao dịch được gán nhãn gian lận.")
        c3.metric("Tỷ lệ bắt sót (Recall)", f"{metrics['recall']:.2%}", help="Khả năng phát hiện đúng và bao phủ tổng lượng giao dịch gian lận thực tế.")
        c4.metric("F1-Score (Cân bằng)", f"{metrics['f1']:.2%}")
        
        st.divider()
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.write("#### Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
            
            # Trực quan hóa ma trận nhầm lẫn bằng sơ đồ nhiệt Plotly
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Dự Đoán (Predicted)", y="Nhãn Thực Tế (Actual)"),
                x=['An toàn (0)', 'Gian lận (1)'],
                y=['An toàn (0)', 'Gian lận (1)'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_res2:
            st.write("#### Tầm quan trọng của các đặc trưng (Feature Importance)")
            model = st.session_state['trained_model']
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Đặc trưng': st.session_state['features_list'],
                'Độ quan trọng': importances
            }).sort_values(by='Độ quan trọng', ascending=True)
            
            fig_imp = px.bar(
                feat_imp_df, x='Độ quan trọng', y='Đặc trưng',
                orientation='h', color='Độ quan trọng',
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: ỨNG DỤNG DỰ BÁO RỦI RO (SỬ DỤNG MÔ HÌNH)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Phân tách luồng kiểm tra dữ liệu giao dịch")
    
    if 'trained_model' not in st.session_state:
        st.info("⚠️ Chức năng này yêu cầu mô hình phải được huấn luyện trước. "
                "Vui lòng nhấn nút '🚀 Huấn luyện mô hình' ở Sidebar bên trái.")
    else:
        model = st.session_state['trained_model']
        
        mode = st.radio(
            "Phương thức nhập dữ liệu kiểm thử:",
            options=["Chế độ 1: Nhập trực tiếp từ đơn lẻ", "Chế độ 2: Tải file dữ liệu kiểm tra hàng loạt (Batch Prediction)"],
            horizontal=True
        )
        
        # ----------------------------------------------------------------------
        # CHẾ ĐỘ 1: NHẬP TRỰC TIẾP
        # ----------------------------------------------------------------------
        if "Chế độ 1" in mode:
            st.write("#### Nhập thông số giao dịch cần chấm điểm rủi ro")
            
            # Lấy giá trị min, max, median từ dữ liệu thô ban đầu để làm giá trị mặc định chuẩn xác
            with st.form("single_prediction_form"):
                st.write("Cấu hình phân bổ biến số đặc trưng:")
                
                # Tạo lưới widget nhập liệu tự động dựa trên danh sách cột
                form_cols = st.columns(3)
                input_data = {}
                
                for idx, feat in enumerate(expected_features):
                    col_widget = form_cols[idx % 3]
                    min_v = float(df_raw[feat].min())
                    max_v = float(df_raw[feat].max())
                    med_v = float(df_raw[feat].median())
                    
                    with col_widget:
                        input_data[feat] = st.number_input(
                            f"Thông số {feat}",
                            min_value=min_v, max_value=max_v, value=med_v,
                            format="%.6f"
                        )
                
                submit_pred = st.form_submit_button("🔍 Thẩm định giao dịch", type="primary")
                
                if submit_pred:
                    # Chuyển đổi dữ liệu sang định dạng DataFrame tương thích
                    single_df = pd.DataFrame([input_data])
                    
                    # Tiến hành dự báo nhãn và tính xác suất
                    prediction = model.predict(single_df)[0]
                    prob = model.predict_proba(single_df)[0][1]
                    
                    st.markdown("### Kết quả phân tích rủi ro:")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        if prediction == 1:
                            st.error("🚨 CẢNH BÁO: Giao dịch có dấu hiệu Gian lận/Rủi ro!")
                        else:
                            st.success("✅ AN TOÀN: Giao dịch nằm trong ngưỡng bình thường.")
                    with col_p2:
                        st.metric("Xác suất phân loại rủi ro", f"{prob:.2%}")
                        
        # ----------------------------------------------------------------------
        # CHẾ ĐỘ 2: TẢI FILE KIỂM TRA HÀNG LOẠT
        # ----------------------------------------------------------------------
        else:
            st.write("#### Tải lên tệp danh sách hồ sơ cần quét rủi ro")
            st.caption("Yêu cầu file tải lên định dạng CSV/Excel chứa đầy đủ các cột đặc trưng từ X_1 đến X_14.")
            
            batch_file = st.file_uploader("Tải tệp kiểm tra hàng loạt", type=["csv", "xlsx"], key="batch_uploader")
            
            if batch_file is not None:
                df_batch = load_data(batch_file.getvalue(), batch_file.name)
                
                if df_batch is not None:
                    # Kiểm tra tính đồng bộ của tập tính năng đầu vào
                    batch_missing = [col for col in expected_features if col not in df_batch.columns]
                    
                    if batch_missing:
                        st.error(f"Cấu trúc file không khớp. File của bạn thiếu các trường thông tin sau: {batch_missing}")
                    else:
                        # Thực hiện dự báo hàng loạt
                        X_batch = df_batch[expected_features]
                        batch_preds = model.predict(X_batch)
                        batch_probas = model.predict_proba(X_batch)[:, 1]
                        
                        # Gán kết quả trực tiếp vào DataFrame mới
                        df_results = df_batch.copy()
                        df_results['Dự báo nhãn (Prediction)'] = batch_preds
                        df_results['Xác suất rủi ro (Risk Probability)'] = batch_probas
                        
                        st.write("### Kết quả chấm điểm hồ sơ hàng loạt")
                        
                        # Đếm thống kê tổng hợp nhanh
                        total_cases = len(df_results)
                        fraud_cases = int(np.sum(batch_preds == 1))
                        
                        st.metric("Tổng số hồ sơ đã quét", f"{total_cases} hồ sơ", 
                                  delta=f"{fraud_cases} trường hợp rủi ro", delta_color="inverse")
                        
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Cho phép xuất dữ liệu kết quả phân tích dưới dạng CSV ký tự UTF-8-SIG để đọc excel tiếng việt
                        csv_buffer = io.StringIO()
                        df_results.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
                        
                        st.download_button(
                            label="📥 Tải xuống báo cáo phân tích rủi ro (.CSV)",
                            data=csv_bytes,
                            file_name="bao_cao_phan_tich_rui_ro_giao_dich.csv",
                            mime="text/csv"
                        )

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
# TAB 1: TỔNG QUAN DỮ LIỆU (HỘP MÀU QUAN SÁT)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Phân tích cấu trúc file dữ liệu")
    file_size_mb = len(file_bytes) / (1024 * 1024)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
            <div style="background-color: #ECEFF1; padding: 20px; border-radius: 10px; border-left: 5px solid #455A64; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                <p style="color: #263238; margin: 0; font-size: 13px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Số lượng quan sát</p>
                <p style="color: #37474F; margin: 5px 0 0 0; font-size: 26px; font-weight: bold;">{df_raw.shape[0]:,}</p>
                <p style="color: #607D8B; margin: 5px 0 0 0; font-size: 12px;">Tổng số hàng (Dòng) dữ liệu</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown(f"""
            <div style="background-color: #E8EAF6; padding: 20px; border-radius: 10px; border-left: 5px solid #3F51B5; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                <p style="color: #1A237E; margin: 0; font-size: 13px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Số lượng biến số</p>
                <p style="color: #283593; margin: 5px 0 0 0; font-size: 26px; font-weight: bold;">{df_raw.shape[1]:,}</p>
                <p style="color: #3F51B5; margin: 5px 0 0 0; font-size: 12px;">Tổng số cột đặc trưng cấu trúc</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        st.markdown(f"""
            <div style="background-color: #E0F2F1; padding: 20px; border-radius: 10px; border-left: 5px solid #009688; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                <p style="color: #004D40; margin: 0; font-size: 13px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Dung lượng tệp tin</p>
                <p style="color: #00695C; margin: 5px 0 0 0; font-size: 26px; font-weight: bold;">{file_size_mb:.2f} MB</p>
                <p style="color: #009688; margin: 5px 0 0 0; font-size: 12px;">Trọng lượng file đã tải lên</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("<br>", unsafe_allow_html=True)
    st.write("### Trích xuất 5 bản ghi dữ liệu đầu tiên (Raw Data)")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.write("### Thống kê mô tả đặc trưng hình học mô hình (X & y)")
    st.dataframe(df_raw[expected_features + [target_col]].describe().T, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA BIẾN CHỈ BÁO (ĐÃ KHẮC PHỤC LỖI THAM SỐ PLOTLY)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Phân tích phân phối biểu đồ tĩnh và động")
    
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
    
    st.write("#### 2. Biểu đồ đường mật độ phân phối mịn (KDE Phân Tích Rủi Ro)")
    st.caption("Gợi ý: Sử dụng biểu đồ đường cong mật độ (KDE) giúp phòng quản trị rủi ro nhận biết ngay phân khúc giao dịch bất thường khi dải màu đỏ (Rủi ro) lệch xa dải màu xanh (An toàn).")
    
    selected_features = st.multiselect(
        "Chọn các biến đặc trưng muốn trực quan hóa phân phối dữ liệu:",
        options=expected_features,
        default=expected_features[:4],
        max_selections=8
    )
    
    if selected_features:
        cols = st.columns(2)
        df_plot = df_raw.copy()
        df_plot['Trạng thái'] = df_plot[target_col].astype(str).map({'0': 'An toàn (0)', '1': 'Rủi ro (1)'})
        
        for idx, feat in enumerate(selected_features):
            col_target = cols[idx % 2]
            with col_target:
                # FIX: Loại bỏ 'element="step"' bị xung đột, sử dụng mô hình overlay phân phối chuẩn
                fig_feat = px.histogram(
                    df_plot, x=feat, color='Trạng thái',
                    marginal="box", 
                    histnorm="probability density", 
                    barmode="overlay",
                    title=f"Mật độ phân bổ đặc trưng {feat} theo phân lớp rủi ro",
                    color_discrete_map={'An toàn (0)': '#1E88E5', 'Rủi ro (1)': '#E53935'} 
                )
                
                # Cấu hình đường viền mượt mờ và độ rộng thanh để giả lập đường KDE liên tục cao cấp
                fig_feat.update_traces(opacity=0.45, marker_line_width=1.5, marker_line_color="white") 
                fig_feat.update_layout(
                    height=380, 
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title=f"Giá trị biến đặc trưng {feat}",
                    yaxis_title="Mật độ phân bổ (Density)"
                )
                st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Vui lòng chọn ít nhất một biến chỉ báo đặc trưng để vẽ biểu đồ.")

# ------------------------------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Chỉ số hiệu năng mô hình phân loại nhị phân")
    
    if 'trained_model' not in st.session_state:
        st.info("⚠️ Mô hình hiện tại chưa được kích hoạt huấn luyện trên cấu hình này. "
                "Vui lòng nhấn nút '🚀 Huấn luyện mô hình' ở Sidebar bên trái để xem kết quả kiểm định.")
    else:
        metrics = st.session_state['evaluation_metrics']
        
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
                <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                    <p style="color: #0D47A1; margin: 0; font-size: 14px; font-weight: bold; text-transform: uppercase;">Accuracy</p>
                    <p style="color: #1565C0; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;">{metrics['accuracy']:.2%}</p>
                    <p style="color: #555; margin: 5px 0 0 0; font-size: 12px;">Độ chính xác tổng thể</p>
                </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
                <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #43A047; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                    <p style="color: #1B5E20; margin: 0; font-size: 14px; font-weight: bold; text-transform: uppercase;">Precision</p>
                    <p style="color: #2E7D32; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;">{metrics['precision']:.2%}</p>
                    <p style="color: #555; margin: 5px 0 0 0; font-size: 12px;">Độ chính xác mô hình</p>
                </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
                <div style="background-color: #FFF3E0; padding: 20px; border-radius: 10px; border-left: 5px solid #FB8C00; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                    <p style="color: #E65100; margin: 0; font-size: 14px; font-weight: bold; text-transform: uppercase;">Recall</p>
                    <p style="color: #EF6C00; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;">{metrics['recall']:.2%}</p>
                    <p style="color: #555; margin: 5px 0 0 0; font-size: 12px;">Tỷ lệ bắt sót rủi ro</p>
                </div>
            """, unsafe_allow_html=True)
            
        with c4:
            st.markdown(f"""
                <div style="background-color: #F3E5F5; padding: 20px; border-radius: 10px; border-left: 5px solid #8E24AA; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                    <p style="color: #4A148C; margin: 0; font-size: 14px; font-weight: bold; text-transform: uppercase;">F1-Score</p>
                    <p style="color: #6A1B9A; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;">{metrics['f1']:.2%}</p>
                    <p style="color: #555; margin: 5px 0 0 0; font-size: 12px;">Chỉ số cân bằng</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.write("#### Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
            
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
# TAB 4: ỨNG DỤNG DỰ BÁO RỦI RO
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Phân tách luồng kiểm tra dữ liệu giao dịch")
    
    if 'trained_model' not in st.session_state:
        st.info("⚠️ Chức năng này yêu cầu mô hình phải được huấn luyện trước. "
                "Vui lòng nhấn nút '🚀 Huấn luyện mô hình' ở Sidebar bên trái.")
    else:
        model = st.session_state['trained_model']
        
        st.write("**Chọn phương thức nhập dữ liệu kiểm thử:**")
        mode = st.segmented_control(
            "Phương thức nhập dữ liệu kiểm thử:",
            options=["📥 Nhập trực tiếp đơn lẻ", "📁 Kiểm tra hàng loạt (Batch)"],
            default="📥 Nhập trực tiếp đơn lẻ",
            label_visibility="collapsed"
        )
        st.write("<br>", unsafe_allow_html=True)
        
        if "Nhập trực tiếp đơn lẻ" in mode:
            st.write("#### Nhập thông số giao dịch cần chấm điểm rủi ro")
            
            with st.form("single_prediction_form"):
                st.write("Cấu hình phân bổ biến số đặc trưng:")
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
                    single_df = pd.DataFrame([input_data])
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
                        
        else:
            st.write("#### Tải lên tệp danh sách hồ sơ cần quét rủi ro")
            st.caption("Yêu cầu file tải lên định dạng CSV/Excel chứa đầy đủ các cột đặc trưng từ X_1 đến X_14.")
            
            batch_file = st.file_uploader("Tải tệp kiểm tra hàng loạt", type=["csv", "xlsx"], key="batch_uploader")
            
            if batch_file is not None:
                df_batch = load_data(batch_file.getvalue(), batch_file.name)
                
                if df_batch is not None:
                    batch_missing = [col for col in expected_features if col not in df_batch.columns]
                    
                    if batch_missing:
                        st.error(f"Cấu trúc file không khớp. File của bạn thiếu các trường thông tin sau: {batch_missing}")
                    else:
                        X_batch = df_batch[expected_features]
                        batch_preds = model.predict(X_batch)
                        batch_probas = model.predict_proba(X_batch)[:, 1]
                        
                        df_results = df_batch.copy()
                        df_results['Dự báo nhãn (Prediction)'] = batch_preds
                        df_results['Xác suất rủi ro (Risk Probability)'] = batch_probas
                        
                        st.write("### Kết quả chấm điểm hồ sơ hàng loạt")
                        total_cases = len(df_results)
                        fraud_cases = int(np.sum(batch_preds == 1))
                        
                        st.metric("Tổng số hồ sơ đã quét", f"{total_cases} hồ sơ", 
                                  delta=f"{fraud_cases} trường hợp rủi ro", delta_color="inverse")
                        
                        st.dataframe(df_results, use_container_width=True)
                        
                        csv_buffer = io.StringIO()
                        df_results.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
                        
                        st.download_button(
                            label="📥 Tải xuống báo cáo phân tích rủi ro (.CSV)",
                            data=csv_bytes,
                            file_name="bao_cao_phan_tich_rui_ro_giao_dich.csv",
                            mime="text/csv"
                        )

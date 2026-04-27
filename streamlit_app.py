# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="财务舞弊预测系统", page_icon="🔍", layout="wide")

st.title("🔍 财务舞弊智能预测系统")
st.markdown("基于机器学习的企业财务风险预警系统")

# 加载模型
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(current_dir, 'results', 'best_model.pkl'))
    scaler = joblib.load(os.path.join(current_dir, 'results', 'scaler.pkl'))
    features = joblib.load(os.path.join(current_dir, 'results', 'feature_names.pkl'))
    
    # 尝试加载SHAP解释器
    try:
        explainer = joblib.load(os.path.join(current_dir, 'results', 'shap_explainer.pkl'))
    except:
        explainer = None
    return model, scaler, features, explainer

with st.spinner('加载模型中...'):
    model, scaler, features, explainer = load_model()

st.success(f"✅ 模型加载成功！使用 {len(features)} 个财务特征")

# ========== 单条预测 ==========
st.markdown("---")
st.subheader("✏️ 单条预测")

with st.expander("手动输入企业财务数据"):
    input_values = {}
    
    # 显示前10个重要特征
    important_features = features[:10]
    cols = st.columns(5)
    for i, feature in enumerate(important_features):
        with cols[i % 5]:
            input_values[feature] = st.number_input(feature, value=0.0, format="%.4f", key=f"single_{feature}")
    
    # 其他特征默认为0
    for feature in features[10:]:
        input_values[feature] = 0.0
    
    if st.button("🔮 单条预测", type="primary"):
        df_single = pd.DataFrame([input_values])[features]
        X_scaled = scaler.transform(df_single)
        
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]
        
        col1, col2 = st.columns(2)
        with col1:
            if pred == 1:
                st.error("⚠️ 预测结果：**舞弊风险高**")
            else:
                st.success("✅ 预测结果：**财务状况正常**")
        with col2:
            st.metric("舞弊概率", f"{prob*100:.1f}%")

# ========== 批量预测 ==========
st.markdown("---")
st.subheader("📁 批量预测")
st.markdown("上传包含财务数据的CSV文件，系统将自动批量预测")

uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 数据预览")
    st.dataframe(df.head())
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"❌ 缺少以下特征列：{missing_features[:5]}...")
    else:
        if st.button("🚀 开始批量预测", type="primary"):
            with st.spinner('预测中...'):
                X = df[features]
                X_scaled = scaler.transform(X)
                
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]
                
                df['预测结果'] = ['舞弊' if p == 1 else '正常' for p in predictions]
                df['舞弊概率'] = probabilities.round(4)
                
                col1, col2, col3 = st.columns(3)
                fraud_count = sum(predictions)
                with col1:
                    st.metric("总样本数", len(df))
                with col2:
                    st.metric("预测舞弊数", fraud_count)
                with col3:
                    st.metric("舞弊比例", f"{fraud_count/len(df)*100:.1f}%")
                
                st.write("### 预测结果详情")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下载预测结果", csv, "预测结果.csv", "text/csv")
                
                # ========== SHAP解释 ==========
                if explainer is not None and st.button("📊 查看SHAP模型解释"):
                    try:
                        import shap
                        shap_values = explainer.shap_values(X_scaled[:5])
                        
                        st.subheader("SHAP特征重要性分析")
                        st.markdown("**红色**：增加舞弊风险 | **蓝色**：降低舞弊风险")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_scaled[:5], feature_names=features, show=False)
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.info(f"SHAP解释生成失败：{e}")
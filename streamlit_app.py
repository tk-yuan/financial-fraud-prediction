# -*- coding: utf-8 -*-
"""
财务舞弊预测系统 - Streamlit版本
运行：streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="财务舞弊智能预测系统",
    page_icon="🔍",
    layout="wide"
)

# 加载模型
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(current_dir, 'results', 'best_model.pkl'))
    scaler = joblib.load(os.path.join(current_dir, 'results', 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(current_dir, 'results', 'feature_names.pkl'))
    return model, scaler, feature_names

# 加载模型
with st.spinner('加载模型中...'):
    model, scaler, feature_names = load_model()

st.title("🔍 财务舞弊智能预测系统")
st.markdown("基于机器学习的企业财务风险预警系统")

# 侧边栏
with st.sidebar:
    st.markdown("## 📊 系统信息")
    st.info(f"模型特征数：{len(feature_names)}")
    st.info("算法：XGBoost / 随机森林")
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("""
    1. **批量预测**：上传CSV文件
    2. **单条预测**：手动输入数据
    """)

# 主界面 - 选项卡
tab1, tab2 = st.tabs(["📁 批量预测", "✏️ 单条预测"])

# ========== 选项卡1：批量预测 ==========
with tab1:
    st.markdown("### 上传企业财务数据CSV文件")
    
    uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"已上传 {len(df)} 条数据")
        st.dataframe(df.head())
        
        if st.button("🚀 开始批量预测", type="primary"):
            with st.spinner('预测中...'):
                try:
                    # 提取特征
                    X = df[feature_names].copy()
                    X_scaled = scaler.transform(X)
                    
                    # 预测
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1]
                    
                    # 添加结果
                    df['预测结果'] = ['舞弊' if p == 1 else '正常' for p in predictions]
                    df['舞弊概率'] = probabilities.round(4)
                    
                    # 显示结果统计
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总样本数", len(df))
                    with col2:
                        fraud_count = (predictions == 1).sum()
                        st.metric("预测舞弊数", fraud_count)
                    with col3:
                        fraud_rate = fraud_count / len(df) * 100
                        st.metric("舞弊比例", f"{fraud_rate:.1f}%")
                    
                    # 显示结果表格
                    st.markdown("### 预测结果详情")
                    st.dataframe(df)
                    
                    # 下载按钮
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下载预测结果",
                        data=csv,
                        file_name='prediction_results.csv',
                        mime='text/csv'
                    )
                    
                except Exception as e:
                    st.error(f"预测失败：{str(e)}")

# ========== 选项卡2：单条预测 ==========
with tab2:
    st.markdown("### 手动输入企业财务数据")
    
    # 使用列布局，让输入更紧凑
    cols_per_row = 4
    feature_groups = [feature_names[i:i+cols_per_row] for i in range(0, len(feature_names), cols_per_row)]
    
    input_values = {}
    
    for group in feature_groups:
        cols = st.columns(cols_per_row)
        for idx, feature in enumerate(group):
            with cols[idx]:
                input_values[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f",
                    key=feature
                )
    
    if st.button("🔮 预测舞弊风险", type="primary"):
        with st.spinner('预测中...'):
            try:
                # 构建DataFrame
                df_single = pd.DataFrame([input_values])[feature_names]
                X_scaled = scaler.transform(df_single)
                
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1]
                
                # 显示结果
                st.markdown("---")
                st.markdown("### 📊 预测结果")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if prediction == 1:
                        st.markdown("## ⚠️")
                        st.error("高风险！可能存在财务舞弊")
                    else:
                        st.markdown("## ✅")
                        st.success("低风险，财务状况正常")
                
                with col2:
                    # 概率仪表盘
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={"text": "舞弊概率 (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "salmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 显示输入的特征值
                with st.expander("查看输入特征值"):
                    st.json(input_values)
                    
            except Exception as e:
                st.error(f"预测失败：{str(e)}")
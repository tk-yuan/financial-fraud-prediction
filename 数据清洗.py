# -*- coding: utf-8 -*-
"""
企业财务舞弊预测 - 数据处理
"""

import pandas as pd
import numpy as np

# 1. 加载数据
violation_data = pd.read_excel(r"D:\Desktop\财务舞弊\违规数据集.xlsx")
financial_data = pd.read_excel(r"D:\Desktop\财务舞弊\财务数据情况.xlsx")

# 2. 定义舞弊标签
fraud_keywords = ['虚假记载', '虚假陈述', '财务造假', '信息披露虚假', 
                  '重大遗漏', '误导性陈述', '业绩预测不准确', '会计差错',
                  '虚构利润', '虚增收入', '财务报告不实','推迟披露',]

def is_financial_fraud(violation_type):
    if pd.isna(violation_type):
        return False
    violation_str = str(violation_type)
    return any(keyword in violation_str for keyword in fraud_keywords)

# 标记舞弊案例
violation_data['is_fraud'] = violation_data['违规类型'].apply(is_financial_fraud)
fraud_cases = violation_data[violation_data['is_fraud'] == True]

# 创建舞弊字典
fraud_dict = {}
for _, row in fraud_cases.iterrows():
    stock_code = str(row['证券代码']).strip()
    fraud_year = row['违规年度']
    
    if pd.isna(fraud_year):
        continue
    try:
        fraud_year = int(fraud_year)
    except:
        continue
    
    if stock_code not in fraud_dict:
        fraud_dict[stock_code] = []
    if fraud_year not in fraud_dict[stock_code]:
        fraud_dict[stock_code].append(fraud_year)

# 3. 为财务数据创建舞弊标签
financial_data['证券代码'] = financial_data['证券代码'].astype(str).str.strip()
financial_data['舞弊标签'] = 0

for idx, row in financial_data.iterrows():
    stock_code = str(row['证券代码']).strip()
    year = row['时间']
    
    if pd.isna(stock_code) or pd.isna(year):
        continue
    
    try:
        year_int = int(year) if isinstance(year, (int, np.integer)) else int(str(year)[:4])
    except:
        continue
    
    if stock_code in fraud_dict:
        fraud_years = fraud_dict[stock_code]
        for fraud_year in fraud_years:
            if abs(year_int - fraud_year) <= 1:
                financial_data.at[idx, '舞弊标签'] = 1
                break

# 4. 数据清洗
# 自动选择所有数值型特征（排除标识列和标签列）
numeric_features = financial_data.select_dtypes(include=[np.number]).columns.tolist()
# 移除非特征列
non_feature_cols = ['证券代码', '时间', '舞弊标签']
features = [col for col in numeric_features if col not in non_feature_cols]

# 选择需要的列
required_cols = ['证券代码', '时间', '舞弊标签'] + features
clean_data = financial_data[required_cols].copy()

# 删除缺失值
clean_data = clean_data.dropna()

# 删除异常值
def remove_outliers_iqr(df, columns):
    rows_to_remove = set()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        rows_to_remove.update(outliers.index.tolist())
    return df.drop(index=list(rows_to_remove))

clean_data = remove_outliers_iqr(clean_data, features)

# 5. 保存最终数据集
clean_data.to_csv('干净数据.csv', index=False, encoding='utf-8-sig')
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 19:25:51 2025

@author: 26819
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_auc_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from xgboost import XGBClassifier

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
import os
data = pd.read_csv(os.path.join('data', 'clean_data.csv'))

print(data.shape)
# 计算正常和舞弊的数量
正常数量 = sum(data['舞弊标签']==0)
舞弊数量 = sum(data['舞弊标签']==1)
总数量 = len(data)

# 计算百分比
正常比例 = 正常数量 / 总数量 * 100
舞弊比例 = 舞弊数量 / 总数量 * 100
print(f"正常样本: {正常数量}个 ({正常比例:.2f}%)")
print(f"舞弊样本: {舞弊数量}个 ({舞弊比例:.2f}%)")

# 分离特征和标签
X = data.drop(['证券代码', '时间', '舞弊标签'], axis=1)
y = data['舞弊标签']

# 2. 数据预处理（按时间顺序划分）
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import numpy as np

# 确保数据按时间排序
data['时间'] = data['时间'].astype(int)
data = data.sort_values('时间').reset_index(drop=True)

# 按时间顺序划分训练集和测试集（前70%训练，后30%测试）
split_ratio = 0.7
split_index = int(len(data) * split_ratio)

# 划分原始数据
train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()
print(f"训练集时间范围: {train_data['时间'].min()} 到 {train_data['时间'].max()}")
print(f"测试集时间范围: {test_data['时间'].min()} 到 {test_data['时间'].max()}")

# 分离特征和标签
X_train_raw = train_data.drop(['证券代码', '时间', '舞弊标签'], axis=1)
y_train = train_data['舞弊标签']
X_test_raw = test_data.drop(['证券代码', '时间', '舞弊标签'], axis=1)
y_test = test_data['舞弊标签']
print(f"\n训练集: {len(X_train_raw)}个样本, 测试集: {len(X_test_raw)}个样本")
print(f"训练集 - 正常样本: {sum(y_train==0)}， 舞弊样本: {sum(y_train==1)}")
print(f"测试集 - 正常样本: {sum(y_test==0)}， 舞弊样本: {sum(y_test==1)}")

# 正确的标准化方法（避免数据泄露）
scaler = StandardScaler()
# 只在训练集上拟合标准化器
X_train = scaler.fit_transform(X_train_raw)
# 用训练集的参数转换测试集
X_test = scaler.transform(X_test_raw)
print(f"\n标准化完成，训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

# 3. SMOTE过采样处理不平衡数据（仅对训练集）
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE后训练集: 正常={sum(y_train_smote==0)}, 舞弊={sum(y_train_smote==1)}")

# 4. 设置交叉验证策略5折
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. 单一模型参数寻优
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# 5.1 决策树参数寻优
dt_param_grid = {
    'max_depth': [7, 15, 21],
    'min_samples_split': [2, 10, 15],
    'min_samples_leaf': [2, 4, 6],
    'criterion': ['gini', 'entropy']
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
dt_grid.fit(X_train_smote, y_train_smote)
dt_model = dt_grid.best_estimator_
print("决策树")
print(f"  最佳参数: {dt_grid.best_params_}")
print(f"  最佳交叉验证AUC: {dt_grid.best_score_:.4f}")

# 5.2 K近邻参数寻优
knn_param_grid = {
    'n_neighbors': [3, 13, 7, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
knn_grid.fit(X_train_smote, y_train_smote)
knn_model = knn_grid.best_estimator_
print("K近邻")
print(f"  最佳参数: {knn_grid.best_params_}")
print(f"  最佳交叉验证AUC: {knn_grid.best_score_:.4f}")

# 5.3 神经网络参数寻优
mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [500]
}
mlp_grid = GridSearchCV(
    MLPClassifier(random_state=42, early_stopping=True),
    mlp_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
mlp_grid.fit(X_train_smote, y_train_smote)
mlp_model = mlp_grid.best_estimator_
print("神经网络")
print(f"  最佳参数: {mlp_grid.best_params_}")
print(f"  最佳交叉验证AUC: {mlp_grid.best_score_:.4f}")

# 6. 集成模型参数寻优
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.ensemble import BalancedBaggingClassifier

# 6.1 随机森林参数寻优
rf_param_grid = {
    'n_estimators': [50, 100, 200, 250],
    'max_depth': [5, 13, 20, 24],
    'min_samples_split': [1, 2, 3, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_smote, y_train_smote)
rf_model = rf_grid.best_estimator_
print("随机森林")
print(f"  最佳参数: {rf_grid.best_params_}")
print(f"  最佳交叉验证AUC: {rf_grid.best_score_:.4f}")

# 6.2 BalancedBagging参数寻优
balanced_bagging_param_grid = {
    'n_estimators': [30, 50, 100, 150],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0],
    'sampling_strategy': [0.5, 0.9],
    'replacement': [False, True]
}

balanced_bagging_grid = GridSearchCV(
    BalancedBaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
        random_state=42
    ),
    balanced_bagging_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
balanced_bagging_grid.fit(X_train, y_train)
balanced_bagging_model = balanced_bagging_grid.best_estimator_
print("BalancedBagging")
print(f"  最佳参数: {balanced_bagging_grid.best_params_}")
print(f"  最佳交叉验证AUC: {balanced_bagging_grid.best_score_:.4f}")

# 6.3 XGBoost参数寻优
from xgboost import XGBClassifier

xgb_param_grid = {
    'n_estimators': [50, 100, 200, 250],
    'max_depth': [3, 9, 11],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 1.0],
    'gamma': [0, 0.1]
}
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    xgb_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
xgb_grid.fit(X_train_smote, y_train_smote)
xgb_model = xgb_grid.best_estimator_
print("xgboost")
print(f"  最佳参数: {xgb_grid.best_params_}")
print(f"  最佳交叉验证AUC: {xgb_grid.best_score_:.4f}")

# 6.4 Voting集成
voting_model = VotingClassifier(
    estimators=[
        ('dt_opt', dt_model),
        ('knn_opt', knn_model),
        ('mlp_opt', mlp_model)
    ],
    voting='soft'
)
voting_model.fit(X_train_smote, y_train_smote)
print("  Voting集成训练完成")

# 7. 保存所有模型
all_models = {
    '决策树': dt_model,
    'K近邻': knn_model,
    '神经网络': mlp_model,
    '随机森林': rf_model,
    'BalancedBagging': balanced_bagging_model,
    'XGBoost': xgb_model,
    'Voting集成': voting_model
}

grid_search_results = {
    '决策树': dt_grid,
    'K近邻': knn_grid,
    '神经网络': mlp_grid,
    '随机森林': rf_grid,
    'BalancedBagging': balanced_bagging_grid,
    'XGBoost': xgb_grid
}

# 8. 模型评估
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_auc_score, roc_curve, auc)

# 获取所有模型的预测结果
models_predictions = {}
for name, model in all_models.items():
    try:
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_prob = y_pred.astype(float)
        models_predictions[name] = (y_pred, y_pred_prob)
    except Exception as e:
        print(f"模型 {name} 预测时出错: {e}")

# 计算各项指标
results = []
confusion_matrices = {}

for name, (y_pred, y_pred_prob) in models_predictions.items():
    try:
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        if len(np.unique(y_pred_prob)) > 1:
            auc_score = roc_auc_score(y_test, y_pred_prob)
        else:
            auc_score = 0.5
        
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            TN, FP, FN, TP = 0, 0, 0, 0
        
        if name in grid_search_results:
            cv_auc_mean = grid_search_results[name].best_score_
        else:
            cv_auc_mean = auc_score
        
        confusion_matrices[name] = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
        results.append([name, acc, precision, recall, f1, auc_score, cv_auc_mean, TP, FP, FN, TN])
        
    except Exception as e:
        print(f"计算模型 {name} 指标时出错: {e}")

# 创建结果DataFrame
results_df = pd.DataFrame(results, columns=['模型', '准确率', '精确率', '召回率', 'F1分数', 
                                           '测试集AUC', '交叉验证AUC', 'TP', 'FP', 'FN', 'TN'])
print(results_df.round(4))


# 9. 可视化结果
# 创建输出目录
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# 9.1 AUC对比图
plt.figure(figsize=(14, 8))
models_for_plot = list(models_predictions.keys())
cv_auc_scores = []
test_auc_scores = []
for model_name in models_for_plot:
    if model_name in grid_search_results:
        cv_auc_scores.append(grid_search_results[model_name].best_score_)
    else:
        cv_auc_scores.append(0)
    
    model_data = results_df[results_df['模型'] == model_name]
    if len(model_data) > 0:
        test_auc_scores.append(model_data['测试集AUC'].values[0])
    else:
        test_auc_scores.append(0)
x = np.arange(len(models_for_plot))
width = 0.35

plt.bar(x - width/2, cv_auc_scores, width, label='交叉验证AUC', alpha=0.8, color='skyblue')
plt.bar(x + width/2, test_auc_scores, width, label='测试集AUC', alpha=0.8, color='lightcoral')
plt.xlabel('模型', fontsize=12)
plt.ylabel('AUC分数', fontsize=12)
plt.title('交叉验证AUC vs 测试集AUC对比', fontsize=14)
plt.xticks(x, models_for_plot, rotation=45, ha='right')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

for i, (cv_auc, test_auc) in enumerate(zip(cv_auc_scores, test_auc_scores)):
    plt.text(i - width/2, cv_auc + 0.01, f'{cv_auc:.3f}', ha='center', fontsize=9)
    plt.text(i + width/2, test_auc + 0.01, f'{test_auc:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/AUC对比图.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.2 最佳模型混淆矩阵
best_model_name = results_df.loc[results_df['测试集AUC'].idxmax(), '模型']
print(f"最佳模型: {best_model_name}")

if best_model_name in confusion_matrices:
    plt.figure(figsize=(10, 8))
    
    cm_data = confusion_matrices[best_model_name]
    TP, FP, FN, TN = cm_data['TP'], cm_data['FP'], cm_data['FN'], cm_data['TN']
    
    cm_array = np.array([[TP, FP], [FN, TN]])
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': '样本数'})
    plt.title(f'最佳模型 ({best_model_name}) 混淆矩阵', fontsize=14)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.xticks([0.5, 1.5], ['正常', '舞弊'])
    plt.yticks([0.5, 1.5], ['正常', '舞弊'])
    
    best_model_row = results_df[results_df['模型'] == best_model_name].iloc[0]
    info_text = f"准确率: {best_model_row['准确率']:.4f}\n精确率: {best_model_row['精确率']:.4f}\n召回率: {best_model_row['召回率']:.4f}\nF1分数: {best_model_row['F1分数']:.4f}\n测试集AUC: {best_model_row['测试集AUC']:.4f}"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'results/figures/最佳模型混淆矩阵_{best_model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 9.3 其他模型混淆矩阵组合图
other_models = [m for m in models_for_plot if m != best_model_name][:4]
if len(other_models) > 0:
    if len(other_models) <= 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    
    for idx, model_name in enumerate(other_models):
        if model_name in confusion_matrices and idx < len(axes):
            cm_data = confusion_matrices[model_name]
            cm_array = np.array([[cm_data['TP'], cm_data['FP']], 
                                 [cm_data['FN'], cm_data['TN']]])
            
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='YlOrRd', 
                       cbar=True, ax=axes[idx], 
                       cbar_kws={'label': '样本数', 'shrink': 0.8})
            
            if model_name in results_df['模型'].values:
                model_perf = results_df[results_df['模型'] == model_name].iloc[0]
                
            axes[idx].set_title(f'{model_name}\nAUC: {model_perf["测试集AUC"]:.3f}, F1: {model_perf["F1分数"]:.3f}', 
                              fontsize=12, pad=15)
            axes[idx].set_xlabel('预测类别', fontsize=10)
            axes[idx].set_ylabel('真实类别', fontsize=10)
            axes[idx].set_xticklabels(['正常', '舞弊'], fontsize=9)
            axes[idx].set_yticklabels(['正常', '舞弊'], fontsize=9, rotation=0)

    for idx in range(len(other_models), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('其他模型混淆矩阵对比 (排除最佳模型)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/其他模型混淆矩阵组合图.png', dpi=300, bbox_inches='tight')
    plt.show()

# 9.4 ROC曲线图
plt.figure(figsize=(12, 10))
colors = plt.cm.tab10(np.linspace(0, 1, len(models_predictions)))
for idx, (name, (_, y_pred_prob)) in enumerate(models_predictions.items()):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[idx], lw=2, 
             label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='随机分类器')
plt.xlabel('假正率 (FPR)', fontsize=12)
plt.ylabel('真正率 (TPR)', fontsize=12)
plt.title('财务舞弊预测 - ROC曲线对比', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

plt.tight_layout()
plt.savefig('results/figures/ROC曲线图.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 前两个模型的特征重要性分析
feature_names = X_train_raw.columns.tolist()

# 按测试集AUC从高到低排序所有模型
sorted_models = results_df.sort_values('测试集AUC', ascending=False)['模型'].tolist()

# 寻找前两个具有特征重要性属性的模型
models_to_analyze = []
for model_name in sorted_models:
    if model_name in all_models:
        model = all_models[model_name]
        try:
            if hasattr(model, 'feature_importances_'):
                models_to_analyze.append(model_name)
                print(f"{model_name}: 具有feature_importances_属性")
                
                if len(models_to_analyze) >= 2:
                    break  # 找到2个就停止
            else:
                print(f"{model_name}: 没有feature_importances_属性")
        except:
            print(f"{model_name}: 检查属性时出错")

if len(models_to_analyze) >= 1:
    for i, model_name in enumerate(models_to_analyze):
        model = all_models[model_name]
        try:
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'特征': feature_names, '重要性': importances})
            importance_df = importance_df.sort_values('重要性', ascending=False)
            
            print(f"\n{model_name}特征重要性排名（前15）:")
            print(importance_df.head(15).round(4))
            
            # 可视化特征重要性
            figure_num = 5 + i
            plt.figure(figsize=(14, 10))
            top_n = min(15, len(importance_df))
            
            colors = ['steelblue', 'darkgreen', 'darkorange', 'purple']
            color_idx = i % len(colors)
            
            plt.barh(range(top_n), importance_df['重要性'].head(top_n), color=colors[color_idx])
            plt.yticks(range(top_n), importance_df['特征'].head(top_n), fontsize=10)
            plt.xlabel('特征重要性', fontsize=12)
            plt.title(f'{model_name}特征重要性分析 (前{top_n}名)', fontsize=14)
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            
            # 添加模型性能信息
            if model_name in results_df['模型'].values:
                model_perf = results_df[results_df['模型'] == model_name].iloc[0]
                info_text = f"测试集AUC: {model_perf['测试集AUC']:.4f}\nF1分数: {model_perf['F1分数']:.4f}\n召回率: {model_perf['召回率']:.4f}\n排名: 第{i+1}位"
                plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'results/figures/特征重要性_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 保存特征重要性数据
            importance_df.to_csv(f'results/特征重要性_{model_name}.csv', index=False, encoding='utf-8-sig')
          
        except Exception as e:
            print(f"{model_name}特征重要性分析失败: {e}")
else:
    print("没有找到具有特征重要性属性的模型，跳过")

# 保存性能结果
results_df.to_csv('results/模型性能结果.csv', index=False, encoding='utf-8-sig')
predictions_data = []
for name, (y_pred, y_pred_prob) in models_predictions.items():
    for i in range(len(y_test)):
        predictions_data.append({
            '模型': name,
            '样本索引': i,
            '真实标签': y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i],
            '预测标签': y_pred[i],
            '预测概率': y_pred_prob[i]
        })

predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv('results/模型预测结果.csv', index=False, encoding='utf-8-sig')

# 保存参数配置信息
param_config = {
    '决策树参数': str(dt_param_grid),
    'K近邻参数': str(knn_param_grid),
    '神经网络参数': str(mlp_param_grid),
    '随机森林参数': str(rf_param_grid),
    'BalancedBagging参数': str(balanced_bagging_param_grid),
    'XGBoost参数': str(xgb_param_grid),
    '时间分割比例': split_ratio,
    '训练集年份范围': f"{train_data['时间'].min()}-{train_data['时间'].max()}",
    '测试集年份范围': f"{test_data['时间'].min()}-{test_data['时间'].max()}"
}

pd.DataFrame([param_config]).to_csv('results/参数配置信息.csv', index=False, encoding='utf-8-sig')
print(f"   - 所有结果已保存到 results/ 目录")


# ========== 保存模型本身（新增）==========
import joblib
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建results文件夹（如果不存在）
os.makedirs(os.path.join(current_dir, 'results'), exist_ok=True)

# 1. 找出最佳模型
best_model_name = results_df.loc[results_df['测试集AUC'].idxmax(), '模型']
best_model = all_models[best_model_name]

# 2. 保存最佳模型
joblib.dump(best_model, os.path.join(current_dir, 'results', 'best_model.pkl'))
print(f"✅ 模型已保存：{best_model_name}")

# 3. 保存标准化器
joblib.dump(scaler, os.path.join(current_dir, 'results', 'scaler.pkl'))
print(f"✅ 标准化器已保存")

# 4. 保存特征列名
feature_names_list = X_train_raw.columns.tolist()
joblib.dump(feature_names_list, os.path.join(current_dir, 'results', 'feature_names.pkl'))
print(f"✅ 特征列名已保存，共{len(feature_names_list)}个")


# ========== 保存SHAP解释器（新增）==========
# ========== 保存SHAP解释器
try:
    import shap
    print("正在生成SHAP解释器...")
    
    # 新版shap的TreeExplainer不需要传background
    explainer = shap.TreeExplainer(best_model)
    
    # 保存解释器
    shap_path = os.path.join(current_dir, 'results', 'shap_explainer.pkl')
    joblib.dump(explainer, shap_path)
    print(f"✅ SHAP解释器已保存到 {shap_path}")
    
except Exception as e:
    print(f"⚠️ SHAP解释器保存失败：{e}")
    print("   （不影响模型使用，SHAP功能将不可用）")
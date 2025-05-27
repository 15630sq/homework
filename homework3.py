import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.dpi"] = 100  # 提升图片清晰度



class Experiment:
    def __init__(self):
        self.datasets = {}
        self.models = {
            '决策树': DecisionTreeClassifier(random_state=42),
            '随机森林': RandomForestClassifier(random_state=42),
            '支持向量机': SVC(probability=True, random_state=42)
        }
        self.feature_selection_methods = {
            '全部特征': None,
            'ANOVA': SelectKBest(score_func=f_classif, k=10),  # 选择10个最佳特征
            'RFE': RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=10)
        }
        self.results = {}

    def load_datasets(self):
        #加载所有实验数据集
        # 加载Iris数据集
        iris = datasets.load_iris()
        X_iris, y_iris = iris.data, iris.target
        self.datasets['Iris'] = (X_iris, y_iris)

        # 加载Breast Cancer数据集
        cancer = datasets.load_breast_cancer()
        X_cancer, y_cancer = cancer.data, cancer.target
        self.datasets['Breast Cancer'] = (X_cancer, y_cancer)

        # 加载THUCNews数据集(使用20 Newsgroups作为替代)
        newsgroups = fetch_20newsgroups(subset='all',
                                        categories=['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'])
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_news = vectorizer.fit_transform(newsgroups.data).toarray()
        y_news = newsgroups.target
        self.datasets['THUCNews替代'] = (X_news, y_news)

        print("数据集加载完成:")
        for name, (X, y) in self.datasets.items():
            print(f"- {name}: 样本数={X.shape[0]}, 特征数={X.shape[1]}, 类别数={len(np.unique(y))}")

    def run_experiment(self):
        #运行完整实验
        self.load_datasets()

        for dataset_name, (X, y) in self.datasets.items():
            print(f"\n正在处理数据集: {dataset_name}")
            self.results[dataset_name] = {}

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # 数据标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for fs_name, fs_method in self.feature_selection_methods.items():
                print(f"  使用特征选择方法: {fs_name}")
                self.results[dataset_name][fs_name] = {}

                # 应用特征选择
                if fs_method is not None:
                    X_train_fs = fs_method.fit_transform(X_train_scaled, y_train)
                    X_test_fs = fs_method.transform(X_test_scaled)
                else:
                    X_train_fs, X_test_fs = X_train_scaled, X_test_scaled

                for model_name, model in self.models.items():
                    print(f"    评估模型: {model_name}")
                    start_time = time.time()

                    # 训练模型
                    model.fit(X_train_fs, y_train)

                    # 预测
                    y_pred = model.predict(X_test_fs)

                    # 计算性能指标
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    # 计算ROC AUC (二分类和多分类处理不同)
                    if len(np.unique(y)) == 2:  # 二分类
                        y_score = model.predict_proba(X_test_fs)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_score)
                    else:  # 多分类
                        y_test_bin = label_binarize(y_test, classes=np.unique(y))
                        y_score = model.predict_proba(X_test_fs)
                        roc_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='weighted')

                    # 计算执行时间
                    execution_time = time.time() - start_time

                    # 保存结果
                    self.results[dataset_name][fs_name][model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'roc_auc': roc_auc,
                        'execution_time': execution_time
                    }

    def visualize_results(self):
        #可视化实验结果
        metrics = ['precision', 'recall', 'f1', 'roc_auc', 'execution_time']
        metric_names = ['精确率', '召回率', 'F1分数', 'ROC AUC', '执行时间(秒)']

        for dataset_name in self.datasets.keys():
            plt.figure(figsize=(18, 12))

            for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                plt.subplot(2, 3, i + 1)

                for fs_name in self.feature_selection_methods.keys():
                    results = self.results[dataset_name][fs_name]
                    models = list(results.keys())
                    values = [results[model][metric] for model in models]

                    plt.bar([x + i * 0.2 for x in range(len(models))], values, width=0.2, label=fs_name)

                plt.title(f'{dataset_name}数据集 - {metric_name}比较')
                plt.xticks([x + 0.2 for x in range(len(models))], models)
                plt.legend()

            plt.tight_layout()
            plt.savefig(f'{dataset_name}_metrics_comparison.png')
            plt.close()

            # 绘制ROC曲线(仅二分类数据集)
            if dataset_name == 'Breast Cancer':
                plt.figure(figsize=(10, 8))
                X, y = self.datasets[dataset_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                for model_name, model in self.models.items():
                    model.fit(X_train_scaled, y_train)
                    y_score = model.predict_proba(X_test_scaled)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = roc_auc_score(y_test, y_score)

                    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假正例率')
                plt.ylabel('真正例率')
                plt.title(f'{dataset_name}数据集 - ROC曲线比较')
                plt.legend(loc="lower right")
                plt.savefig(f'{dataset_name}_roc_comparison.png')
                plt.close()

    def print_results(self):
        #打印实验结果
        print("\n\n===== 实验结果 =====")

        for dataset_name in self.datasets.keys():
            print(f"\n3.1 数据集: {dataset_name}")

            for fs_name in self.feature_selection_methods.keys():
                print(f"\n  特征选择方法: {fs_name}")
                print("  模型\t\t精确率\t\t召回率\t\tF1分数\t\tROC AUC\t\t执行时间(秒)")
                print("  --------------------------------------------------------------------------------")

                for model_name, result in self.results[dataset_name][fs_name].items():
                    print(
                        f"  {model_name}\t{result['precision']:.4f}\t\t{result['recall']:.4f}\t\t{result['f1']:.4f}\t\t{result['roc_auc']:.4f}\t\t{result['execution_time']:.4f}")


if __name__ == "__main__":
    experiment = Experiment()
    experiment.run_experiment()
    experiment.visualize_results()
    experiment.print_results()
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Flatten, Dense, Reshape, LeakyReLU, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import tensorflow as tf

warnings.filterwarnings('ignore')


# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_and_preprocess(dataset_name='emnist_digits', sample_size=5000):
    """加载并预处理数据集"""
    if dataset_name == 'emnist_digits':
        dataset = tfds.load('emnist/digits', split='train+test', as_supervised=True)
        x = np.array([x.numpy() for x, y in dataset])[:sample_size]
        y = np.array([y.numpy() for x, y in dataset])[:sample_size]
        x = x.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    elif dataset_name == 'cifar10':
        dataset = tfds.load('cifar10', split='train+test', as_supervised=True)
        x = np.array([x.numpy() for x, y in dataset])[:sample_size]
        y = np.array([y.numpy() for x, y in dataset])[:sample_size]
        x = x.reshape(-1, 32, 32, 3).astype('float32') / 255.0
    else:
        raise ValueError("支持'emnist_digits'或'cifar10'")

    # 划分训练集和测试集
    split_idx = int(len(x) * 0.8)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 数据验证
    print(f"数据形状: 训练集={x_train.shape}, 测试集={x_test.shape}")
    print(f"标签范围: {np.min(y_train)}~{np.max(y_train)}")

    return x_train, y_train, x_test, y_test


# ---------------------------
# 2. 改进的卷积自编码器
# ---------------------------
def build_advanced_cae(input_shape, encoding_dim=32):
    """构建带尺寸匹配的卷积自编码器"""
    input_img = Input(shape=input_shape)

    # 编码器
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 14x14 or 16x16

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 7x7 or 8x8

    # 根据输入尺寸调整
    if input_shape[0] == 28:  # EMNIST
        pool_shape = (7, 7)
    else:  # CIFAR-10
        pool_shape = (8, 8)

    x = Flatten()(x)
    encoded = Dense(encoding_dim, activation='relu')(x)

    # 解码器
    x = Dense(64 * pool_shape[0] * pool_shape[1], activation='relu')(encoded)
    x = Reshape((pool_shape[0], pool_shape[1], 64))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

    cae = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    cae.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return cae, encoder


# ---------------------------
# 3. 训练自编码器并提取特征
# ---------------------------
def train_and_extract_features(x_train, x_test, encoding_dim=32, epochs=30, batch_size=64):
    """训练CAE并提取特征"""
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"{'使用GPU加速' if gpu_available else '使用CPU'}")

    try:
        cae, encoder = build_advanced_cae(x_train.shape[1:], encoding_dim)

        # 添加早停机制
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # 训练模型
        history = cae.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, x_test),
            callbacks=[early_stopping],
            verbose=1 if gpu_available else 2
        )

        # 提取特征
        train_features = encoder.predict(x_train, batch_size=batch_size)
        test_features = encoder.predict(x_test, batch_size=batch_size)

        # 生成重建图像
        reconstructions = cae.predict(x_test, batch_size=batch_size)

        return train_features, test_features, encoder, history, reconstructions

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        return None, None, None, None, None


# ---------------------------
# 4. 改进的聚类评估
# ---------------------------
def evaluate_clustering(features, true_labels, dataset_name, feature_type, n_clusters=10, encoding_dim=None):
    """执行K-means聚类并计算评估指标"""
    if features is None or len(features) == 0:
        print(f"警告: {feature_type}特征为空，跳过聚类评估")
        return {
            'silhouette': -1,
            'ari': -1,
            'pred_labels': np.array([]),
            'confusion_matrix': np.array([]),
            'features_reduced': np.array([]),
            'encoding_dim': encoding_dim
        }

    # 确保特征在聚类前被归一化
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)

    # 对于高维特征使用PCA降维
    if features_norm.shape[1] > 100:
        pca = PCA(n_components=min(100, features_norm.shape[1]))
        features_reduced = pca.fit_transform(features_norm)
        print(f"  原始维度 {features.shape[1]} -> PCA降维到 {features_reduced.shape[1]}")
    else:
        features_reduced = features_norm

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    pred_labels = kmeans.fit_predict(features_reduced)

    # 计算评估指标
    silhouette = silhouette_score(features_reduced, pred_labels) if n_clusters > 1 else -1
    ari = adjusted_rand_score(true_labels, pred_labels)

    # 归一化混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(f"\n{dataset_name} - {feature_type}特征{'' if encoding_dim is None else f' (编码维度={encoding_dim})'}:")
    print(f"  轮廓系数: {silhouette:.4f} | 调整兰德指数: {ari:.4f}")

    return {
        'silhouette': silhouette,
        'ari': ari,
        'pred_labels': pred_labels,
        'confusion_matrix': cm_normalized,
        'features_reduced': features_reduced,
        'encoding_dim': encoding_dim
    }


# ---------------------------
# 5. 改进的可视化分析
# ---------------------------
def visualize_results(x_test, ae_features, raw_features, true_labels, ae_results, raw_results,
                      reconstructions, dataset_name, ae_dim_results=None):
    """可视化t-SNE降维和性能对比"""
    if x_test is None or len(x_test) == 0:
        print("警告: 无测试数据可供可视化")
        return

    plt.figure(figsize=(20, 16))

    # 1. 原始图像与重建图像对比
    plt.subplot(3, 4, 1)
    visualize_reconstructions(x_test, reconstructions, true_labels, ae_results['pred_labels'],
                              dataset_name, n_samples=5)

    # 2. AE特征t-SNE（按真实标签着色）
    plt.subplot(3, 4, 2)
    visualize_tsne(ae_results['features_reduced'], true_labels, "AE特征 (真实标签)", dataset_name)

    # 3. AE特征t-SNE（按聚类结果着色）
    plt.subplot(3, 4, 3)
    visualize_tsne(ae_results['features_reduced'], ae_results['pred_labels'],
                   "AE特征 (聚类结果)", dataset_name)

    # 4. 原始特征t-SNE（按真实标签着色）
    plt.subplot(3, 4, 5)
    visualize_tsne(raw_results['features_reduced'], true_labels, "原始特征 (真实标签)", dataset_name)

    # 5. 原始特征t-SNE（按聚类结果着色）
    plt.subplot(3, 4, 6)
    visualize_tsne(raw_results['features_reduced'], raw_results['pred_labels'],
                   "原始特征 (聚类结果)", dataset_name)

    # 6. 混淆矩阵热力图
    plt.subplot(3, 4, 4)
    plot_confusion_matrix(ae_results['confusion_matrix'], "AE特征混淆矩阵")

    # 7. 易混淆样本分析
    plt.subplot(3, 4, 7)
    analyze_confused_samples(x_test, true_labels, ae_results['pred_labels'],
                             ae_results['features_reduced'], dataset_name, top_n=2)

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_extended_comparison.png")
    plt.close()

    # 性能对比条形图
    visualize_performance_comparison(ae_results, raw_results, dataset_name)

    # 编码维度性能对比
    if ae_dim_results:
        visualize_dimension_comparison(ae_dim_results, dataset_name)


# ---------------------------
# 6. 改进的辅助可视化函数
# ---------------------------
def visualize_reconstructions(x_test, reconstructions, true_labels, pred_labels,
                              dataset_name, n_samples=5):
    """可视化原始图像和重建图像"""
    if x_test is None or len(x_test) == 0:
        return plt.figure()

    fig, axes = plt.subplots(2, n_samples, figsize=(15, 4))

    for i in range(min(n_samples, len(x_test))):
        # 原始图像
        if x_test.shape[-1] == 1:  # 灰度图
            axes[0, i].imshow(x_test[i].squeeze(), cmap='gray')
        else:  # 彩色图
            axes[0, i].imshow(x_test[i])
        axes[0, i].set_title(f"原始\n真实: {true_labels[i]}\n聚类: {pred_labels[i]}")
        axes[0, i].axis('off')

        # 重建图像
        if reconstructions.shape[-1] == 1:  # 灰度图
            axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        else:  # 彩色图
            axes[1, i].imshow(reconstructions[i])
        axes[1, i].set_title("重建")
        axes[1, i].axis('off')

    plt.suptitle(f"{dataset_name} - 原始图像与重建图像对比")
    return fig


def visualize_tsne(features, labels, title, dataset_name):
    """执行t-SNE降维并可视化"""
    if features is None or len(features) == 0:
        print(f"警告: 无特征数据可供t-SNE降维 ({title})")
        return

    # 对于大数据集进行子采样以加速t-SNE
    if len(features) > 2000:
        indices = np.random.choice(len(features), 2000, replace=False)
        features = features[indices]
        labels = labels[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(features)

    plt.scatter(tsne_features[:, 0], tsne_features[:, 1],
                c=labels, cmap='tab10', alpha=0.8, s=30)
    plt.title(title)
    plt.colorbar()


def plot_confusion_matrix(cm, title):
    """绘制归一化混淆矩阵热力图"""
    if cm is None or len(cm) == 0:
        print(f"警告: 混淆矩阵为空 ({title})")
        return

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()


def visualize_performance_comparison(ae_results, raw_results, dataset_name):
    """可视化AE特征与原始特征的性能对比"""
    metrics = ['轮廓系数', '调整兰德指数']
    ae_values = [ae_results['silhouette'], ae_results['ari']]
    raw_values = [raw_results['silhouette'], raw_results['ari']]

    # 处理轮廓系数为负数的情况
    ae_values = [max(0, v) for v in ae_values]
    raw_values = [max(0, v) for v in raw_values]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, ae_values, width, label='AE特征')
    rects2 = ax.bar(x + width / 2, raw_values, width, label='原始特征')

    ax.set_ylabel('分数')
    ax.set_title(f'{dataset_name} - AE特征 vs 原始特征的聚类性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)  # 统一Y轴范围

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.4f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(f"{dataset_name}_performance.png")
    plt.close()


# ---------------------------
# 7. 新增：参数调优可视化函数
# ---------------------------
def visualize_dimension_comparison(dim_results, dataset_name):
    """可视化不同编码维度的聚类性能"""
    if not dim_results:
        print("警告: 无编码维度结果可供可视化")
        return

    dims = sorted(dim_results.keys())
    silhouettes = [dim_results[dim]['silhouette'] for dim in dims]
    aris = [dim_results[dim]['ari'] for dim in dims]

    plt.figure(figsize=(12, 6))

    # 轮廓系数对比
    plt.subplot(1, 2, 1)
    plt.plot(dims, silhouettes, marker='o', color='tab:blue')
    plt.title(f'{dataset_name} - 编码维度 vs 轮廓系数')
    plt.xlabel('编码维度')
    plt.ylabel('轮廓系数')
    plt.grid(True)

    # 调整兰德指数对比
    plt.subplot(1, 2, 2)
    plt.plot(dims, aris, marker='o', color='tab:orange')
    plt.title(f'{dataset_name} - 编码维度 vs 调整兰德指数')
    plt.xlabel('编码维度')
    plt.ylabel('ARI')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_dimension_performance.png")
    plt.close()


# ---------------------------
# 8. 新增：错误样本分析函数
# ---------------------------
def analyze_confused_samples(x_test, true_labels, pred_labels, ae_features_reduced, dataset_name, top_n=5):
    """分析易混淆样本并可视化"""
    if x_test is None or len(x_test) == 0:
        print("警告: 无测试数据可供分析")
        return

    cm = confusion_matrix(true_labels, pred_labels)
    np.fill_diagonal(cm, 0)  # 忽略正确分类

    # 计算类别间错误率
    confused = {}
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                key = tuple(sorted((i, j)))
                confused[key] = confused.get(key, 0) + cm[i, j]

    # 按错误次数排序
    sorted_confused = sorted(confused.items(), key=lambda x: x[1], reverse=True)[:top_n]

    plt.figure(figsize=(15, 8))
    plt.suptitle(f"{dataset_name} - 易混淆样本分析", y=1.02)

    for idx, (pair, count) in enumerate(sorted_confused):
        i, j = pair

        # 提取混淆样本索引
        mask_i = (true_labels == i) & (pred_labels == j)
        mask_j = (true_labels == j) & (pred_labels == i)
        indices_i = np.where(mask_i)[0][:2]  # 每个类别取2个样本
        indices_j = np.where(mask_j)[0][:2]

        # 绘制混淆样本
        for sub_idx, idx_ in enumerate(indices_i):
            if sub_idx < 2 and idx_ < len(x_test):
                plt.subplot(top_n, 4, idx * 4 + sub_idx + 1)
                if x_test.shape[-1] == 1:
                    plt.imshow(x_test[idx_].squeeze(), cmap='gray')
                else:
                    plt.imshow(x_test[idx_])
                true_label = true_labels[idx_]
                pred_label = pred_labels[idx_]
                plt.title(f"T:{true_label}\nP:{pred_label}")
                plt.axis('off')

        for sub_idx, idx_ in enumerate(indices_j):
            if sub_idx < 2 and idx_ < len(x_test):
                plt.subplot(top_n, 4, idx * 4 + sub_idx + 3)
                if x_test.shape[-1] == 1:
                    plt.imshow(x_test[idx_].squeeze(), cmap='gray')
                else:
                    plt.imshow(x_test[idx_])
                true_label = true_labels[idx_]
                pred_label = pred_labels[idx_]
                plt.title(f"T:{true_label}\nP:{pred_label}")
                plt.axis('off')

    # 绘制t-SNE全局图（仅在有特征数据时）
    if ae_features_reduced is not None and len(ae_features_reduced) > 0:
        plt.subplot(top_n, 4, 4)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_features = tsne.fit_transform(ae_features_reduced)
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1],
                    c=true_labels, cmap='tab10', alpha=0.6, s=20)
        plt.title("全局t-SNE分布")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_confused_samples.png")
    plt.close()


# ---------------------------
# 9. 新增：易混淆类别分析
# ---------------------------
def analyze_confused_classes(true_labels, pred_labels, class_names=None):
    """分析易混淆的类别对"""
    cm = confusion_matrix(true_labels, pred_labels)
    np.fill_diagonal(cm, 0)  # 忽略正确分类

    # 获取错误次数最多的类别对
    confused_pairs = []
    for i in range(cm.shape[0]):
        for j in range(i + 1, cm.shape[1]):
            if cm[i, j] > 0 or cm[j, i] > 0:
                error_count = cm[i, j] + cm[j, i]
                confused_pairs.append((i, j, error_count))

    # 按错误次数排序
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    # 打印Top5易混淆类别对
    print("\n易混淆的类别对:")
    for i, j, count in confused_pairs[:5]:
        class_i = class_names[i] if class_names else str(i)
        class_j = class_names[j] if class_names else str(j)
        total = np.sum(true_labels == i) + np.sum(true_labels == j)
        error_rate = count / total
        print(f"- {class_i} 和 {class_j}: 错误次数={count}, 错误率={error_rate:.4f}")

    return confused_pairs


# ---------------------------
# 10. 主函数
# ---------------------------
def main():
    # 配置参数
    DATASET = 'emnist_digits'  # 可选 'emnist_digits' 或 'cifar10'
    SAMPLE_SIZE = 5000  # 样本量
    ENCODING_DIMS = [16, 32, 64, 128, 256]  # 待测试的编码维度
    EPOCHS = 30  # 训练轮次
    BATCH_SIZE = 128  # 批次大小

    # 加载数据
    x_train, y_train, x_test, y_test = load_and_preprocess(DATASET, SAMPLE_SIZE)
    print(f"成功加载{DATASET}数据集")
    print(f"训练集: {len(x_train)}, 测试集: {len(x_test)}")

    # 展平原始特征（用于对比）
    raw_train_features = x_train.reshape(len(x_train), -1)
    raw_test_features = x_test.reshape(len(x_test), -1)
    print(f"原始特征维度: {raw_train_features.shape[1]}")

    # ---------------------------
    # 参数调优：测试不同编码维度
    # ---------------------------
    ae_dim_results = {}
    best_ari = -1
    best_dim = None
    best_ae_results = None
    best_ae_features = None

    for dim in ENCODING_DIMS:
        print(f"\n=== 训练编码维度 {dim} ===")
        train_features, test_features, _, _, reconstructions = train_and_extract_features(
            x_train, x_test, dim, EPOCHS, BATCH_SIZE
        )

        # 检查特征提取是否成功
        if test_features is None or len(test_features) == 0:
            print(f"警告: 编码维度 {dim} 的特征提取失败，跳过此维度")
            continue

        # 评估AE特征的聚类性能
        ae_result = evaluate_clustering(test_features, y_test, DATASET, "AE",
                                        n_clusters=10, encoding_dim=dim)
        ae_dim_results[dim] = ae_result

        # 记录最优维度（基于ARI）
        if ae_result['ari'] > best_ari:
            best_ari = ae_result['ari']
            best_dim = dim
            best_ae_features = test_features
            best_ae_results = ae_result

    if best_ae_results is None:
        print("错误: 所有编码维度的特征提取均失败，程序终止")
        return

    print(f"\n最优编码维度: {best_dim} (ARI={best_ari:.4f})")

    # ---------------------------
    # 评估原始特征的聚类性能
    # ---------------------------
    print("\n评估原始特征的聚类性能...")
    raw_results = evaluate_clustering(raw_test_features, y_test, DATASET, "原始")

    # ---------------------------
    # 可视化对比结果
    # ---------------------------
    visualize_results(x_test, best_ae_features, raw_test_features, y_test,
                      best_ae_results, raw_results, reconstructions, DATASET,
                      ae_dim_results=ae_dim_results)

    # ---------------------------
    # 输出易混淆类别分析
    # ---------------------------
    print("\n易混淆类别分析:")
    analyze_confused_classes(y_test, best_ae_results['pred_labels'])

    # ---------------------------
    # 保存最优维度的训练历史
    # ---------------------------
    print(f"\n重新训练最优维度 {best_dim} 以保存训练历史...")
    _, _, _, history, _ = train_and_extract_features(
        x_train, x_test, best_dim, EPOCHS, BATCH_SIZE
    )
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title(f'最优维度 {best_dim} 训练历史')
    plt.ylabel('损失')
    plt.xlabel('训练轮次')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{DATASET}_best_dim_training.png')
    plt.close()

    print("\n分析完成！生成的文件:")
    print(f"- 维度性能对比: {DATASET}_dimension_performance.png")
    print(f"- 扩展对比分析: {DATASET}_extended_comparison.png")
    print(f"- 易混淆样本图: {DATASET}_confused_samples.png")
    print(f"- 性能对比: {DATASET}_performance.png")
    print(f"- 最优维度训练历史: {DATASET}_best_dim_training.png")


if __name__ == "__main__":
    main()
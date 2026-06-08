"""
欧洲银行客户分群数据处理脚本（增强版）
Bank Customer Segmentation Data Processing & Clustering Script

功能:
- 数据加载与探索
- 数据清洗与预处理
- 特征工程
- 数据标准化
- 手动PCA降维
- 手肘法确定最优K值
- K-Means聚类分析
- 结果可视化与统计
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ManualPCA:
    """手动实现的PCA降维类"""
    
    def __init__(self, n_components=2):
        """
        初始化PCA
        
        Args:
            n_components (int): 降维后的维数
        """
        self.n_components = n_components
        self.mean = None
        self.covariance_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.components = None
        self.explained_variance_ratio = None
        self.cumulative_variance_ratio = None
        
    def fit(self, X):
        """
        训练PCA模型
        
        Args:
            X (ndarray): 输入数据 (n_samples, n_features)
        """
        print("\n" + "=" * 60)
        print("🔧 手动PCA - 拟合过程")
        print("=" * 60)
        
        # 步骤1: 中心化数据
        print("\n[步骤 1/5] 中心化数据...")
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        print(f"  ✓ 均值: {self.mean[:5]}... (显示前5个)")
        
        # 步骤2: 计算协方差矩阵
        print("\n[步骤 2/5] 计算协方差矩阵...")
        n_samples = X.shape[0]
        self.covariance_matrix = np.cov(X_centered.T)
        print(f"  ✓ 协方差矩阵形状: {self.covariance_matrix.shape}")
        print(f"  ✓ 协方差矩阵对角线元素: {np.diag(self.covariance_matrix)[:5]}... (前5个)")
        
        # 步骤3: 计算特征值和特征向量
        print("\n[步骤 3/5] 计算特征值和特征向量...")
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance_matrix)
        
        # 按特征值从大到小排序
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        
        print(f"  ✓ 特征值: {self.eigenvalues[:5]}... (前5个)")
        print(f"  ✓ 特征值总和: {np.sum(self.eigenvalues):.4f}")
        
        # 步骤4: 计算方差解释比例
        print("\n[步骤 4/5] 计算方差解释比例...")
        self.explained_variance_ratio = self.eigenvalues / np.sum(self.eigenvalues)
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)
        
        print(f"  ✓ 方差贡献率 (前5个): {self.explained_variance_ratio[:5]}")
        print(f"  ✓ 累计方差贡献率 (前5个): {self.cumulative_variance_ratio[:5]}")
        
        # 步骤5: 选择主成分
        print(f"\n[步骤 5/5] 选择前 {self.n_components} 个主成分...")
        self.components = self.eigenvectors[:, :self.n_components].real
        print(f"  ✓ 主成分矩阵形状: {self.components.shape}")
        print(f"  ✓ 累计方差贡献率: {self.cumulative_variance_ratio[self.n_components-1]:.4f} "
              f"({self.cumulative_variance_ratio[self.n_components-1]*100:.2f}%)")
        
        return self
    
    def transform(self, X):
        """
        降维数据
        
        Args:
            X (ndarray): 输入数据
            
        Returns:
            ndarray: 降维后的数据
        """
        X_centered = X - self.mean
        X_transformed = np.dot(X_centered, self.components)
        return X_transformed
    
    def fit_transform(self, X):
        """拟合并降维数据"""
        return self.fit(X).transform(X)


class ElbowMethodAnalyzer:
    """手肘法分析类"""
    
    def __init__(self, X, k_range=(2, 11)):
        """
        初始化手肘法分析器
        
        Args:
            X (ndarray): 数据
            k_range (tuple): K值范围 (min, max)
        """
        self.X = X
        self.k_range = range(k_range[0], k_range[1])
        self.inertias = []
        self.silhouette_scores = []
        self.optimal_k = None
        
    def analyze(self):
        """执行手肘法分析"""
        print("\n" + "=" * 60)
        print("📊 手肘法分析")
        print("=" * 60)
        
        from sklearn.metrics import silhouette_score
        
        print(f"\n分析K值范围: {self.k_range.start} - {self.k_range.stop - 1}")
        print("\nK值\t内聚度(惯性)\t轮廓系数")
        print("-" * 50)
        
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X)
            inertia = kmeans.inertia_
            silhouette = silhouette_score(self.X, kmeans.labels_)
            
            self.inertias.append(inertia)
            self.silhouette_scores.append(silhouette)
            
            print(f"{k}\t{inertia:.2f}\t\t{silhouette:.4f}")
        
        # 根据轮廓系数找到最优K值
        self.optimal_k = self.k_range.start + np.argmax(self.silhouette_scores)
        print(f"\n✓ 最优K值 (基于轮廓系数): {self.optimal_k}")
        print(f"  轮廓系数: {max(self.silhouette_scores):.4f}")
        
        return self.optimal_k
    
    def plot_elbow_curve(self, save_path='elbow_curve.png'):
        """绘制手肘法曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: 内聚度曲线
        axes[0].plot(self.k_range, self.inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].axvline(x=self.optimal_k, color='r', linestyle='--', linewidth=2, label=f'最优K={self.optimal_k}')
        axes[0].set_xlabel('K值', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('内聚度 (Inertia)', fontsize=12, fontweight='bold')
        axes[0].set_title('手肘法 - 内聚度曲线', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_xticks(self.k_range)
        
        # 右图: 轮廓系数曲线
        axes[1].plot(self.k_range, self.silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].axvline(x=self.optimal_k, color='r', linestyle='--', linewidth=2, label=f'最优K={self.optimal_k}')
        axes[1].set_xlabel('K值', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('轮廓系数 (Silhouette Score)', fontsize=12, fontweight='bold')
        axes[1].set_title('手肘法 - 轮廓系数曲线', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_xticks(self.k_range)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 手肘法曲线已保存: {save_path}")
        plt.close()


class ClusteringAnalyzer:
    """聚类分析类"""
    
    def __init__(self, X, n_clusters=3):
        """
        初始化聚类分析器
        
        Args:
            X (ndarray): 输入数据
            n_clusters (int): 聚类数
        """
        self.X = X
        self.n_clusters = n_clusters
        self.kmeans = None
        self.labels = None
        self.cluster_centers = None
        
    def fit(self):
        """执行K-Means聚类"""
        print("\n" + "=" * 60)
        print("🎯 K-Means聚类分析")
        print("=" * 60)
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.X)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        print(f"\n✓ K-Means模型拟合完成")
        print(f"  - 聚类数: {self.n_clusters}")
        print(f"  - 内聚度: {self.kmeans.inertia_:.4f}")
        
        # 计算各簇的样本数
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\n各簇样本数分布:")
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(self.labels)) * 100
            print(f"  - 簇 {cluster_id}: {count} 个样本 ({percentage:.2f}%)")
        
        return self
    
    def get_cluster_statistics(self, original_data, feature_names):
        """
        获取各簇的统计信息
        
        Args:
            original_data (DataFrame): 原始数据
            feature_names (list): 特征名列表
            
        Returns:
            dict: 聚类统计结果
        """
        print("\n" + "=" * 60)
        print("📈 各簇特征统计")
        print("=" * 60)
        
        stats = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels == cluster_id
            cluster_data = original_data[cluster_mask]
            
            print(f"\n【簇 {cluster_id}】 (样本数: {cluster_mask.sum()})")
            print("-" * 60)
            
            cluster_stats = {}
            for col in original_data.columns:
                if original_data[col].dtype in [np.float64, np.int64]:
                    mean_val = cluster_data[col].mean()
                    std_val = cluster_data[col].std()
                    min_val = cluster_data[col].min()
                    max_val = cluster_data[col].max()
                    
                    cluster_stats[col] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val
                    }
                    
                    print(f"  {col:25s}: 均值={mean_val:10.2f}, 标准差={std_val:8.2f}, "
                          f"范围=[{min_val:10.2f}, {max_val:10.2f}]")
            
            stats[f'Cluster_{cluster_id}'] = cluster_stats
        
        return stats
    
    def plot_clusters_2d(self, X_pca, save_path='clusters_2d.png'):
        """
        绘制2D聚类可视化散点图
        
        Args:
            X_pca (ndarray): PCA降维后的数据
            save_path (str): 保存路径
        """
        print("\n🎨 生成2D聚类散点图...")
        plt.figure(figsize=(12, 8))
        
        # 定义颜色和标记
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FF8C94', '#A8D8EA']
        markers = ['o', 's', '^', 'D', 'v', 'p', '*']
        
        # 绘制各簇的数据点
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[cluster_id % len(colors)],
                       marker=markers[cluster_id % len(markers)],
                       s=80, alpha=0.7, label=f'簇 {cluster_id}',
                       edgecolors='black', linewidth=0.5)
        
        plt.xlabel(f'PC1 (第一主成分)', fontsize=12, fontweight='bold')
        plt.ylabel(f'PC2 (第二主成分)', fontsize=12, fontweight='bold')
        plt.title(f'K-Means聚类结果 - 2D散点图 (K={self.n_clusters})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 2D聚类散点图已保存: {save_path}")
        plt.close()
    
    def plot_clusters_3d(self, X_pca, save_path='clusters_3d.png'):
        """
        绘制3D聚类可视化散点图
        
        Args:
            X_pca (ndarray): PCA降维后的数据
            save_path (str): 保存路径
        """
        if X_pca.shape[1] < 3:
            print("⚠️  数据维度不足3维，跳过3D可视化")
            return
        
        print("\n🎨 生成3D聚类散点图...")
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FF8C94', '#A8D8EA']
        markers = ['o', 's', '^', 'D', 'v', 'p', '*']
        
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                      c=colors[cluster_id % len(colors)],
                      marker=markers[cluster_id % len(markers)],
                      s=80, alpha=0.7, label=f'簇 {cluster_id}',
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
        ax.set_zlabel('PC3', fontsize=11, fontweight='bold')
        ax.set_title(f'K-Means聚类结果 - 3D散点图 (K={self.n_clusters})', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 3D聚类散点图已保存: {save_path}")
        plt.close()
    
    def plot_cluster_distribution(self, save_path='cluster_distribution.png'):
        """
        绘制聚类分布柱状图
        
        Args:
            save_path (str): 保存路径
        """
        print("\n🎨 生成聚类分布柱状图...")
        unique, counts = np.unique(self.labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FF8C94', '#A8D8EA']
        bars = ax.bar([f'簇 {i}' for i in unique], counts, 
                      color=[colors[i % len(colors)] for i in unique],
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / len(self.labels)) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('聚类簇', fontsize=12, fontweight='bold')
        ax.set_ylabel('样本数量', fontsize=12, fontweight='bold')
        ax.set_title(f'各簇样本分布 (总样本数: {len(self.labels)})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 聚类分布柱状图已保存: {save_path}")
        plt.close()


class BankCustomerDataProcessor:
    """银行客户数据处理类（增强版）"""
    
    def __init__(self, file_path):
        """
        初始化数据处理器
        
        Args:
            file_path (str): CSV文件路径
        """
        self.file_path = file_path
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.pca = None
        self.X_pca = None
        self.clustering_results = {}
        
    def load_data(self):
        """加载数据"""
        print("=" * 60)
        print("📊 第一步: 加载数据")
        print("=" * 60)
        
        self.df = pd.read_csv(self.file_path)
        print(f"✓ 数据加载成功")
        print(f"  - 数据形状: {self.df.shape}")
        print(f"  - 行数: {self.df.shape[0]}, 列数: {self.df.shape[1]}")
        
        return self.df
    
    def explore_data(self):
        """数据探索"""
        print("\n" + "=" * 60)
        print("🔍 第二步: 数据探索")
        print("=" * 60)
        
        print("\n📌 数据基本信息:")
        print(self.df.info())
        
        print("\n📊 前5行数据:")
        print(self.df.head())
        
        print("\n📈 统计描述:")
        print(self.df.describe())
        
        print("\n❌ 缺失值检查:")
        missing_count = self.df.isnull().sum()
        if missing_count.sum() == 0:
            print("  ✓ 没有缺失值")
        else:
            print(missing_count[missing_count > 0])
        
        return self.df
    
    def clean_data(self):
        """数据清洗"""
        print("\n" + "=" * 60)
        print("🧹 第三步: 数据清洗")
        print("=" * 60)
        
        self.df_processed = self.df.copy()
        
        # 1. 删除不需要的列
        print("\n1️⃣  删除不相关列...")
        columns_to_drop = ['CustomerId', 'Surname']
        self.df_processed = self.df_processed.drop(columns=columns_to_drop, errors='ignore')
        print(f"  ✓ 删除了 {len(columns_to_drop)} 列")
        
        # 2. 处理缺失值
        print("\n2️⃣  处理缺失值...")
        if self.df_processed.isnull().sum().sum() == 0:
            print("  ✓ 没有缺失值需要处理")
        else:
            self.df_processed = self.df_processed.dropna()
            print(f"  ✓ 删除了包含缺失值的行")
        
        # 3. 删除重复值
        print("\n3️⃣  处理重复值...")
        duplicates = self.df_processed.duplicated().sum()
        if duplicates > 0:
            self.df_processed = self.df_processed.drop_duplicates()
            print(f"  ✓ 删除了 {duplicates} 行重复数据")
        else:
            print("  ✓ 没有重复值")
        
        print(f"\n清洗后数据形状: {self.df_processed.shape}")
        return self.df_processed
    
    def feature_engineering(self):
        """特征工程"""
        print("\n" + "=" * 60)
        print("⚙️  第四步: 特征工程")
        print("=" * 60)
        
        # 1. 分类变量转换
        print("\n1️⃣  处理分类变量...")
        
        geography_dummies = pd.get_dummies(self.df_processed['Geography'], 
                                          prefix='Geography', 
                                          drop_first=False)
        self.df_processed = pd.concat([self.df_processed, geography_dummies], axis=1)
        print(f"  ✓ Geography 转换完成 -> {geography_dummies.shape[1]} 个特征")
        
        gender_encoder = LabelEncoder()
        self.df_processed['Gender_Encoded'] = gender_encoder.fit_transform(self.df_processed['Gender'])
        self.label_encoders['Gender'] = gender_encoder
        print(f"  ✓ Gender 编码完成 (0=Female, 1=Male)")
        
        # 2. 创建新特征
        print("\n2️⃣  创建衍生特征...")
        
        self.df_processed['Customer_Value_Index'] = (
            self.df_processed['CreditScore'] / 1000 + 
            self.df_processed['Balance'] / 100000 + 
            self.df_processed['EstimatedSalary'] / 100000
        )
        print("  ✓ 客户价值指数 (Customer_Value_Index)")
        
        self.df_processed['Account_Activity'] = (
            self.df_processed['IsActiveMember'] * self.df_processed['NumOfProducts']
        )
        print("  ✓ 账户活跃度 (Account_Activity)")
        
        def credit_score_level(score):
            if score < 500:
                return 'Poor'
            elif score < 650:
                return 'Fair'
            elif score < 750:
                return 'Good'
            else:
                return 'Excellent'
        
        self.df_processed['Credit_Score_Level'] = self.df_processed['CreditScore'].apply(credit_score_level)
        print("  ✓ 信用评分等级 (Credit_Score_Level)")
        
        def age_group(age):
            if age < 30:
                return 'Young'
            elif age < 40:
                return 'Middle'
            elif age < 50:
                return 'Senior'
            else:
                return 'Elderly'
        
        self.df_processed['Age_Group'] = self.df_processed['Age'].apply(age_group)
        print("  ✓ 年龄分组 (Age_Group)")
        
        def balance_level(balance):
            if balance == 0:
                return 'No_Balance'
            elif balance < 50000:
                return 'Low'
            elif balance < 150000:
                return 'Medium'
            else:
                return 'High'
        
        self.df_processed['Balance_Level'] = self.df_processed['Balance'].apply(balance_level)
        print("  ✓ 账户余额等级 (Balance_Level)")
        
        print(f"\n特征工程后列数: {self.df_processed.shape[1]}")
        return self.df_processed
    
    def prepare_for_clustering(self):
        """为聚类准备数据"""
        print("\n" + "=" * 60)
        print("🎯 第五步: 为聚类准备数据")
        print("=" * 60)
        
        clustering_features = [
            'CreditScore',
            'Age',
            'Tenure',
            'Balance',
            'NumOfProducts',
            'HasCrCard',
            'IsActiveMember',
            'EstimatedSalary',
            'Gender_Encoded',
            'Customer_Value_Index',
            'Account_Activity'
        ]
        
        geography_cols = [col for col in self.df_processed.columns if col.startswith('Geography_')]
        clustering_features.extend(geography_cols)
        
        self.feature_columns = clustering_features
        
        print(f"\n选择的聚类特征 ({len(clustering_features)} 个):")
        for i, feat in enumerate(clustering_features, 1):
            print(f"  {i:2d}. {feat}")
        
        X = self.df_processed[clustering_features].copy()
        
        print(f"\n✓ 特征矩阵形状: {X.shape}")
        
        return X
    
    def normalize_data(self, X):
        """数据标准化"""
        print("\n" + "=" * 60)
        print("📏 第六步: 数据标准化 (Standardization)")
        print("=" * 60)
        
        X_normalized = self.scaler.fit_transform(X)
        print("✓ 数据标准化完成")
        print(f"  - 方法: StandardScaler (均值=0, 标准差=1)")
        print(f"  - 输入形状: {X.shape}")
        print(f"  - 输出形状: {X_normalized.shape}")
        
        print(f"\n标准化后数据统计:")
        print(f"  - 均值: {X_normalized.mean():.6f}")
        print(f"  - 标准差: {X_normalized.std():.6f}")
        print(f"  - 最小值: {X_normalized.min():.4f}")
        print(f"  - 最大值: {X_normalized.max():.4f}")
        
        return X_normalized
    
    def apply_manual_pca(self, X_normalized, n_components=2):
        """应用手动PCA降维"""
        print("\n" + "=" * 60)
        print("🔬 第七步: 手动PCA降维")
        print("=" * 60)
        
        self.pca = ManualPCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(X_normalized)
        
        print(f"\n✓ PCA降维完成")
        print(f"  - 原始维度: {X_normalized.shape[1]}")
        print(f"  - 降维后维度: {self.X_pca.shape[1]}")
        print(f"  - 累计方差贡献率: {self.pca.cumulative_variance_ratio[n_components-1]*100:.2f}%")
        
        return self.X_pca
    
    def elbow_method(self, X_normalized, k_range=(2, 11)):
        """执行手肘法分析"""
        print("\n" + "=" * 60)
        print("🔍 第八步: 手肘法分析")
        print("=" * 60)
        
        elbow_analyzer = ElbowMethodAnalyzer(X_normalized, k_range=k_range)
        optimal_k = elbow_analyzer.analyze()
        elbow_analyzer.plot_elbow_curve('elbow_curve.png')
        
        self.clustering_results['optimal_k'] = optimal_k
        self.clustering_results['elbow_analyzer'] = elbow_analyzer
        
        return optimal_k, elbow_analyzer
    
    def perform_clustering(self, X_normalized, optimal_k):
        """执行聚类分析"""
        print("\n" + "=" * 60)
        print("🎨 第九步: 执行K-Means聚类与可视化")
        print("=" * 60)
        
        clustering = ClusteringAnalyzer(X_normalized, n_clusters=optimal_k)
        clustering.fit()
        
        # 获取聚类统计
        cluster_stats = clustering.get_cluster_statistics(
            self.df_processed[self.feature_columns],
            self.feature_columns
        )
        
        # 生成多种聚类可视化
        print("\n生成聚类可视化...")
        
        if self.X_pca.shape[1] >= 2:
            clustering.plot_clusters_2d(self.X_pca, 'clusters_2d_scatter.png')
        
        if self.X_pca.shape[1] >= 3:
            clustering.plot_clusters_3d(self.X_pca, 'clusters_3d_scatter.png')
        
        # 绘制聚类分布
        clustering.plot_cluster_distribution('cluster_distribution.png')
        
        self.clustering_results['clustering'] = clustering
        self.clustering_results['cluster_stats'] = cluster_stats
        self.clustering_results['labels'] = clustering.labels
        
        return clustering, cluster_stats
    
    def save_results(self):
        """保存结果"""
        print("\n" + "=" * 60)
        print("💾 第十步: 保存结果")
        print("=" * 60)
        
        # 1. 保存处理后的数据
        self.df_processed.to_csv('bank_customers_processed.csv', index=False)
        print("✓ 已保存: bank_customers_processed.csv")
        
        # 2. 保存聚类标签
        result_df = self.df_processed.copy()
        result_df['Cluster'] = self.clustering_results['labels']
        result_df.to_csv('bank_customers_with_clusters.csv', index=False)
        print("✓ 已保存: bank_customers_with_clusters.csv")
        
        # 3. 保存方差贡献率报告
        self._save_pca_report()
        
        # 4. 保存聚类统计报告
        self._save_clustering_report()
    
    def _save_pca_report(self):
        """保存PCA报告"""
        with open('pca_variance_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("PCA方差贡献率分析报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("【主成分方差贡献率】\n")
            f.write("-" * 70 + "\n")
            f.write("主成分\t方差贡献率\t累计贡献率\n")
            f.write("-" * 70 + "\n")
            
            for i, (var, cum_var) in enumerate(zip(
                self.pca.explained_variance_ratio[:10],
                self.pca.cumulative_variance_ratio[:10]
            ), 1):
                f.write(f"PC{i}\t{var*100:.4f}%\t\t{cum_var*100:.4f}%\n")
            
            f.write("\n【特征值排序】\n")
            f.write("-" * 70 + "\n")
            for i, eigenvalue in enumerate(self.pca.eigenvalues[:10], 1):
                f.write(f"λ{i} = {eigenvalue:.6f}\n")
            
            f.write(f"\n【降维结果】\n")
            f.write("-" * 70 + "\n")
            f.write(f"原始维度: {len(self.feature_columns)}\n")
            f.write(f"目标维度: {self.X_pca.shape[1]}\n")
            f.write(f"累计方差贡献率: {self.pca.cumulative_variance_ratio[self.X_pca.shape[1]-1]*100:.2f}%\n")
        
        print("✓ 已保存: pca_variance_report.txt")
    
    def _save_clustering_report(self):
        """保存聚类统计报告"""
        with open('clustering_analysis_report.txt', 'w', encoding='utf-8') as f:
            optimal_k = self.clustering_results['optimal_k']
            cluster_stats = self.clustering_results['cluster_stats']
            clustering = self.clustering_results['clustering']
            
            f.write("=" * 70 + "\n")
            f.write("K-Means聚类分析报告\n")
            f.write("=" * 70 + "\n\n")
            
            # 最优K值
            f.write("【最优K值分析】\n")
            f.write("-" * 70 + "\n")
            f.write(f"最优簇数: {optimal_k}\n")
            f.write(f"内聚度: {clustering.kmeans.inertia_:.4f}\n\n")
            
            # 手肘法结果
            elbow = self.clustering_results['elbow_analyzer']
            f.write("【手肘法详细结果】\n")
            f.write("-" * 70 + "\n")
            f.write("K值\t内聚度\t\t轮廓系数\n")
            f.write("-" * 70 + "\n")
            for k, inertia, silhouette in zip(
                elbow.k_range,
                elbow.inertias,
                elbow.silhouette_scores
            ):
                f.write(f"{k}\t{inertia:.4f}\t\t{silhouette:.4f}\n")
            
            # 各簇统计
            f.write(f"\n【各簇特征统计】\n")
            f.write("-" * 70 + "\n")
            
            for cluster_name, stats in cluster_stats.items():
                f.write(f"\n{cluster_name}:\n")
                for feature, metrics in stats.items():
                    f.write(f"  {feature}:\n")
                    f.write(f"    均值: {metrics['mean']:.4f}\n")
                    f.write(f"    标准差: {metrics['std']:.4f}\n")
                    f.write(f"    范围: [{metrics['min']:.4f}, {metrics['max']:.4f}]\n")
        
        print("✓ 已保存: clustering_analysis_report.txt")
    
    def generate_summary(self):
        """生成总结报告"""
        print("\n" + "=" * 60)
        print("📋 数据处理完整总结")
        print("=" * 60)
        
        print(f"\n✅ 数据处理完成!")
        
        print(f"\n【原始数据统计】")
        print(f"  - 原始行数: {self.df.shape[0]}")
        print(f"  - 原始列数: {self.df.shape[1]}")
        
        print(f"\n【清洗后数据统计】")
        print(f"  - 清洗后行数: {self.df_processed.shape[0]}")
        print(f"  - 清洗后列数: {self.df_processed.shape[1]}")
        
        print(f"\n【特征工程统计】")
        print(f"  - 聚类特征数: {len(self.feature_columns)}")
        
        print(f"\n【PCA降维统计】")
        print(f"  - 降维后维度: {self.X_pca.shape[1]}")
        print(f"  - 累计方差贡献率: {self.pca.cumulative_variance_ratio[self.X_pca.shape[1]-1]*100:.2f}%")
        
        print(f"\n【最优聚类结果】")
        optimal_k = self.clustering_results['optimal_k']
        print(f"  - 最优K值: {optimal_k}")
        
        unique, counts = np.unique(self.clustering_results['labels'], return_counts=True)
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(self.clustering_results['labels'])) * 100
            print(f"    - 簇 {cluster_id}: {count} 个样本 ({percentage:.2f}%)")
        
        print(f"\n【输出文件】")
        print(f"  ✓ bank_customers_processed.csv - 处理后的数据")
        print(f"  ✓ bank_customers_with_clusters.csv - 带聚类标签的数据")
        print(f"  ✓ pca_variance_report.txt - PCA方差贡献率报告")
        print(f"  ✓ clustering_analysis_report.txt - 聚类分析报告")
        print(f"  ✓ elbow_curve.png - 手肘法曲线图")
        print(f"  ✓ clusters_2d_scatter.png - 2D聚类散点图 ⭐")
        print(f"  ✓ clusters_3d_scatter.png - 3D聚类散点图 ⭐")
        print(f"  ✓ cluster_distribution.png - 聚类分布柱状图 ⭐")
    
    def run_full_pipeline(self):
        """运行完整数据处理管道"""
        print("\n")
        print("🚀" * 30)
        print("🎯 开始完整数据处理与聚类管道")
        print("🚀" * 30)
        
        # 步骤1-6: 数据处理
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.feature_engineering()
        X = self.prepare_for_clustering()
        X_normalized = self.normalize_data(X)
        
        # 步骤7: PCA降维
        self.apply_manual_pca(X_normalized, n_components=2)
        
        # 步骤8: 手肘法分析
        optimal_k, elbow_analyzer = self.elbow_method(X_normalized, k_range=(2, 11))
        
        # 步骤9: 聚类分析
        clustering, cluster_stats = self.perform_clustering(X_normalized, optimal_k)
        
        # 步骤10: 保存结果
        self.save_results()
        
        # 总结
        self.generate_summary()
        
        print("\n" + "🎉" * 30)
        print("✅ 完整数据处理与聚类管道完成!")
        print("🎉" * 30 + "\n")
        
        return {
            'df_processed': self.df_processed,
            'X_pca': self.X_pca,
            'optimal_k': optimal_k,
            'clustering_labels': clustering.labels,
            'cluster_stats': cluster_stats,
            'pca': self.pca,
            'elbow_analyzer': elbow_analyzer
        }


def main():
    """主函数"""
    processor = BankCustomerDataProcessor('Bank_Churn.csv')
    results = processor.run_full_pipeline()
    return results


if __name__ == '__main__':
    results = main()

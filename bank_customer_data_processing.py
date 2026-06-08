"""
欧洲银行客户分群数据处理脚本
Bank Customer Segmentation Data Processing Script

功能:
- 数据加载与探索
- 数据清洗与预处理
- 特征工程
- 数据标准化
- 用于聚类的数据准备
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BankCustomerDataProcessor:
    """银行客户数据处理类"""
    
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
        
        print("\n🔢 数据类型分布:")
        print(self.df.dtypes)
        
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
        columns_to_drop = ['CustomerId', 'Surname']  # ID和姓名不需要
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
        
        # Geography (地理位置) - One-Hot编码
        geography_dummies = pd.get_dummies(self.df_processed['Geography'], 
                                          prefix='Geography', 
                                          drop_first=False)
        self.df_processed = pd.concat([self.df_processed, geography_dummies], axis=1)
        print(f"  ✓ Geography 转换完成 -> {geography_dummies.shape[1]} 个特征")
        
        # Gender (性别) - Label编码
        gender_encoder = LabelEncoder()
        self.df_processed['Gender_Encoded'] = gender_encoder.fit_transform(self.df_processed['Gender'])
        self.label_encoders['Gender'] = gender_encoder
        print(f"  ✓ Gender 编码完成 (0=Female, 1=Male)")
        
        # 2. 创建新特征
        print("\n2️⃣  创建衍生特征...")
        
        # 客户价值指数 (基于多个因素)
        self.df_processed['Customer_Value_Index'] = (
            self.df_processed['CreditScore'] / 1000 + 
            self.df_processed['Balance'] / 100000 + 
            self.df_processed['EstimatedSalary'] / 100000
        )
        print("  ✓ 客户价值指数 (Customer_Value_Index)")
        
        # 账户活跃度
        self.df_processed['Account_Activity'] = (
            self.df_processed['IsActiveMember'] * self.df_processed['NumOfProducts']
        )
        print("  ✓ 账户活跃度 (Account_Activity)")
        
        # 信用评分等级
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
        
        # 年龄分组
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
        
        # 账户余额等级
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
        
        # 选择用于聚类的特征
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
        
        # 添加地理位置特征
        geography_cols = [col for col in self.df_processed.columns if col.startswith('Geography_')]
        clustering_features.extend(geography_cols)
        
        self.feature_columns = clustering_features
        
        print(f"\n选择的聚类特征 ({len(clustering_features)} 个):")
        for i, feat in enumerate(clustering_features, 1):
            print(f"  {i:2d}. {feat}")
        
        # 提取特征
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
        
        # 显示标准化后的统计
        print(f"\n标准化后数据统计:")
        print(f"  - 均值: {X_normalized.mean():.6f}")
        print(f"  - 标准差: {X_normalized.std():.6f}")
        print(f"  - 最小值: {X_normalized.min():.4f}")
        print(f"  - 最大值: {X_normalized.max():.4f}")
        
        return X_normalized
    
    def perform_eda(self):
        """执行探索性数据分析"""
        print("\n" + "=" * 60)
        print("📊 第七步: 探索性数据分析 (EDA)")
        print("=" * 60)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('银行客户数据分析', fontsize=16, fontweight='bold')
        
        # 1. 年龄分布
        axes[0, 0].hist(self.df_processed['Age'], bins=30, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('年龄分布')
        axes[0, 0].set_xlabel('年龄')
        axes[0, 0].set_ylabel('频数')
        
        # 2. 账户余额分布
        axes[0, 1].hist(self.df_processed['Balance'], bins=30, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('账户余额分布')
        axes[0, 1].set_xlabel('余额')
        axes[0, 1].set_ylabel('频数')
        
        # 3. 信用评分分布
        axes[0, 2].hist(self.df_processed['CreditScore'], bins=30, color='coral', edgecolor='black')
        axes[0, 2].set_title('信用评分分布')
        axes[0, 2].set_xlabel('评分')
        axes[0, 2].set_ylabel('频数')
        
        # 4. 客户流失比例
        churn_counts = self.df_processed['Exited'].value_counts()
        labels = ['保留', '流失']
        axes[1, 0].pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        axes[1, 0].set_title('客户流失比例')
        
        # 5. 按地理位置的客户数
        geo_counts = self.df_processed['Geography'].value_counts()
        axes[1, 1].bar(geo_counts.index, geo_counts.values, color='steelblue', edgecolor='black')
        axes[1, 1].set_title('按地理位置的客户分布')
        axes[1, 1].set_ylabel('客户数')
        
        # 6. 性别分布
        gender_counts = self.df_processed['Gender'].value_counts()
        axes[1, 2].bar(gender_counts.index, gender_counts.values, color='pink', edgecolor='black')
        axes[1, 2].set_title('按性别的客户分布')
        axes[1, 2].set_ylabel('客户数')
        
        plt.tight_layout()
        plt.savefig('bank_customer_eda.png', dpi=300, bbox_inches='tight')
        print("✓ 探索性分析图表已保存: bank_customer_eda.png")
        plt.close()
    
    def generate_summary_report(self, X_normalized):
        """生成数据处理总结报告"""
        print("\n" + "=" * 60)
        print("📋 数据处理总结报告")
        print("=" * 60)
        
        print(f"\n✅ 数据处理完成!")
        print(f"\n原始数据统计:")
        print(f"  - 原始行数: {self.df.shape[0]}")
        print(f"  - 原始列数: {self.df.shape[1]}")
        
        print(f"\n清洗后数据统计:")
        print(f"  - 清洗后行数: {self.df_processed.shape[0]}")
        print(f"  - 清洗后列数: {self.df_processed.shape[1]}")
        
        print(f"\n聚类特征统计:")
        print(f"  - 选择的特征数: {len(self.feature_columns)}")
        print(f"  - 标准化后数据形状: {X_normalized.shape}")
        
        print(f"\n客户分布:")
        print(f"  - 总客户数: {len(self.df_processed)}")
        print(f"  - 流失客户: {self.df_processed['Exited'].sum()} ({self.df_processed['Exited'].mean()*100:.2f}%)")
        print(f"  - 保留客户: {(self.df_processed['Exited']==0).sum()} ({(1-self.df_processed['Exited'].mean())*100:.2f}%)")
        
        print(f"\n数据已准备就绪，可用于:")
        print(f"  ✓ K-Means聚类分析")
        print(f"  ✓ 层次聚类")
        print(f"  ✓ DBSCAN聚类")
        print(f"  ✓ 高斯混合模型 (GMM)")
        print(f"  ✓ 深度学习模型训练")
        
    def run_full_pipeline(self):
        """运行完整数据处理管道"""
        print("\n")
        print("🚀" * 30)
        print("🎯 开始数据处理管道")
        print("🚀" * 30)
        
        # 步骤1: 加载数据
        self.load_data()
        
        # 步骤2: 探索数据
        self.explore_data()
        
        # 步骤3: 清洗数据
        self.clean_data()
        
        # 步骤4: 特征工程
        self.feature_engineering()
        
        # 步骤5: 为聚类准备
        X = self.prepare_for_clustering()
        
        # 步骤6: 数据标准化
        X_normalized = self.normalize_data(X)
        
        # 步骤7: EDA
        self.perform_eda()
        
        # 步骤8: 生成报告
        self.generate_summary_report(X_normalized)
        
        print("\n" + "🎉" * 30)
        print("✅ 数据处理管道完成!")
        print("🎉" * 30 + "\n")
        
        return {
            'df': self.df_processed,
            'X': X,
            'X_normalized': X_normalized,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }


def main():
    """主函数"""
    
    # 初始化处理器
    processor = BankCustomerDataProcessor('Bank_Churn.csv')
    
    # 运行完整管道
    results = processor.run_full_pipeline()
    
    # 保存处理后的数据
    print("\n💾 保存处理后的数据...")
    results['df'].to_csv('bank_customers_processed.csv', index=False)
    print("✓ 已保存: bank_customers_processed.csv")
    
    # 保存标准化的特征矩阵
    X_normalized_df = pd.DataFrame(
        results['X_normalized'],
        columns=results['feature_columns']
    )
    X_normalized_df.to_csv('bank_customers_normalized_features.csv', index=False)
    print("✓ 已保存: bank_customers_normalized_features.csv")
    
    return results


if __name__ == '__main__':
    results = main()

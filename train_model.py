#!/usr/bin/env python3
"""
设备故障预测模型训练脚本
用于训练XGBoost模型并保存到本地
"""

import argparse
from fault_predictor_core import OptimizedDeviceFaultPredictor
from datetime import datetime
import os

# 数据库配置
DB_CONFIG = {
    'host': '117.72.55.146',
    'port': 1012,
    'database': 'postgres',
    'user': 'dbreader',
    'password': 'db@123456'
}

def train_model(args):
    """
    训练故障预测模型
    """
    print("="*60)
    print("设备故障预测模型训练")
    print("="*60)
    
    # 创建预测器实例
    predictor = OptimizedDeviceFaultPredictor(DB_CONFIG)
    
    # 加载训练数据
    print(f"\n加载最近 {args.days} 天的数据...")
    df = predictor.load_data(
        recent_days=args.days,
        limit=args.limit,
        device_id=args.device_id if args.device_id else None
    )
    
    if df is None or len(df) == 0:
        print("错误: 未能加载数据")
        return
    
    # 特征工程
    print("\n进行特征工程...")
    df_features = predictor.enhanced_feature_engineering(df)
    
    # 准备特征
    X, y, feature_cols = predictor.prepare_features(df_features)
    
    print(f"\n数据集信息:")
    print(f"- 样本数: {len(X)}")
    print(f"- 特征数: {len(feature_cols)}")
    print(f"- 故障样本数: {y.sum()}")
    print(f"- 故障率: {y.mean():.4f}")
    
    # 检查故障样本数
    if y.sum() < 5:
        print(f"\n错误: 故障样本太少({y.sum()}个)，至少需要5个故障样本进行训练")
        return
    
    # 训练模型
    print("\n开始训练模型...")
    model = predictor.train_xgboost_model(X, y, use_resampling=args.use_resampling)
    
    if model is None:
        print("错误: 模型训练失败")
        return
    
    # 交叉验证
    if args.validate:
        print("\n进行交叉验证...")
        predictor.validate_model_robustness(X, y, n_splits=args.cv_splits)
    
    # 保存模型
    model_path = args.model_path
    if not model_path:
        # 生成默认模型文件名
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"fault_prediction_model.pkl"
    
    predictor.save_model(model_path)
    
    # 保存特征重要性报告
    if args.save_report:
        report_path = model_path.replace('.pkl', '_feature_importance.csv')
        predictor.feature_importance.to_csv(report_path, index=False)
        print(f"特征重要性报告已保存到 {report_path}")
    
    print(f"\n训练完成！模型已保存到 {model_path}")

def main():
    parser = argparse.ArgumentParser(description='训练设备故障预测模型')
    
    # 数据相关参数
    parser.add_argument('--days', type=int, default=7,
                        help='加载最近多少天的数据进行训练 (默认: 7)')
    parser.add_argument('--limit', type=int, default=100000,
                        help='数据条数限制 (默认: 100000)')
    parser.add_argument('--device-id', type=str, default=None,
                        help='指定设备ID，不指定则加载所有设备数据')
    
    # 模型相关参数
    parser.add_argument('--use-resampling', action='store_true', default=True,
                        help='是否使用SMOTE进行数据重采样 (默认: True)')
    parser.add_argument('--no-resampling', dest='use_resampling', action='store_false',
                        help='不使用数据重采样')
    
    # 验证相关参数
    parser.add_argument('--validate', action='store_true', default=True,
                        help='是否进行交叉验证 (默认: True)')
    parser.add_argument('--cv-splits', type=int, default=5,
                        help='交叉验证的折数 (默认: 5)')
    
    # 输出相关参数
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型保存路径 (默认: 自动生成带时间戳的文件名)')
    parser.add_argument('--save-report', action='store_true', default=True,
                        help='是否保存特征重要性报告 (默认: True)')
    
    args = parser.parse_args()
    
    # 执行训练
    train_model(args)

if __name__ == "__main__":
    main()
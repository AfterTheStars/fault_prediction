#!/usr/bin/env python3
"""
设备故障批量预测脚本
用于批量预测多个设备的故障风险并生成报告
"""

import argparse
from fault_predictor_core import OptimizedDeviceFaultPredictor
import pandas as pd
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

def load_device_list(file_path):
    """
    从文件加载设备ID列表
    支持txt文件（每行一个ID）或csv文件（第一列为ID）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"设备列表文件不存在: {file_path}")
    
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            device_ids = [line.strip() for line in f if line.strip()]
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        device_ids = df.iloc[:, 0].astype(str).tolist()
    else:
        raise ValueError("不支持的文件格式，请使用.txt或.csv文件")
    
    return device_ids

def batch_predict(args):
    """
    批量预测设备故障
    """
    print("="*60)
    print("设备故障批量预测")
    print("="*60)
    
    # 创建预测器实例
    predictor = OptimizedDeviceFaultPredictor(DB_CONFIG)
    
    # 加载模型
    if not predictor.load_model(args.model_path):
        print(f"错误: 无法加载模型文件 {args.model_path}")
        return
    
    # 获取设备列表
    if args.device_file:
        try:
            device_ids = load_device_list(args.device_file)
            print(f"\n从文件加载了 {len(device_ids)} 个设备ID")
        except Exception as e:
            print(f"错误: 无法加载设备列表 - {e}")
            return
    elif args.device_ids:
        device_ids = args.device_ids
        print(f"\n指定了 {len(device_ids)} 个设备ID")
    else:
        # 如果没有指定设备，从数据库获取所有活跃设备
        print("\n未指定设备，将获取所有活跃设备...")
        df_all = predictor.load_data(recent_days=1, limit=10000)
        if df_all is None or len(df_all) == 0:
            print("错误: 无法获取设备列表")
            return
        device_ids = df_all['f_device'].unique().tolist()[:args.max_devices]
        print(f"获取到 {len(device_ids)} 个活跃设备")
    
    # 执行批量预测
    print(f"\n开始批量预测，时间范围: 最近 {args.hours} 小时")
    summary_df = predictor.batch_predict_devices(
        device_ids=device_ids,
        recent_hours=args.hours
    )
    
    if len(summary_df) == 0:
        print("错误: 批量预测失败，没有结果")
        return
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    if args.save_details:
        details_path = f"batch_prediction_details_{timestamp}.csv"
        summary_df.to_csv(details_path, index=False)
        print(f"\n详细结果已保存到 {details_path}")
    
    # 生成汇总报告
    print("\n" + "="*60)
    print("批量预测汇总报告")
    print("="*60)
    
    # 按风险等级分组
    high_risk_devices = summary_df[
        (summary_df['status'] == '正常') & 
        (summary_df['max_fault_probability'] >= args.high_risk_threshold)
    ]
    medium_risk_devices = summary_df[
        (summary_df['status'] == '正常') & 
        (summary_df['max_fault_probability'] >= args.medium_risk_threshold) &
        (summary_df['max_fault_probability'] < args.high_risk_threshold)
    ]
    low_risk_devices = summary_df[
        (summary_df['status'] == '正常') & 
        (summary_df['max_fault_probability'] < args.medium_risk_threshold)
    ]
    
    print(f"\n风险分布:")
    print(f"- 高风险设备 (>{args.high_risk_threshold:.0%}): {len(high_risk_devices)} 个")
    print(f"- 中风险设备 ({args.medium_risk_threshold:.0%}-{args.high_risk_threshold:.0%}): {len(medium_risk_devices)} 个")
    print(f"- 低风险设备 (<{args.medium_risk_threshold:.0%}): {len(low_risk_devices)} 个")
    print(f"- 无数据设备: {len(summary_df[summary_df['status'] == '无数据'])} 个")
    print(f"- 预测失败设备: {len(summary_df[summary_df['status'] == '预测失败'])} 个")
    
    # 显示高风险设备
    if len(high_risk_devices) > 0:
        print("\n高风险设备列表:")
        print("-" * 60)
        for _, device in high_risk_devices.iterrows():
            print(f"设备ID: {device['device_id']}")
            print(f"  - 最大故障概率: {device['max_fault_probability']:.2%}")
            print(f"  - 平均故障概率: {device['avg_fault_probability']:.2%}")
            print(f"  - 故障预测次数: {device['fault_predictions']}")
            print(f"  - 极高风险记录数: {device['extreme_risk_count']}")
            print()
    
    # 保存报告
    if args.save_report:
        report_path = f"batch_prediction_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("设备故障批量预测报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"预测设备数: {len(device_ids)}\n")
            f.write(f"数据时间范围: 最近 {args.hours} 小时\n")
            f.write("="*60 + "\n\n")
            
            f.write("风险分布:\n")
            f.write(f"- 高风险设备: {len(high_risk_devices)} 个\n")
            f.write(f"- 中风险设备: {len(medium_risk_devices)} 个\n")
            f.write(f"- 低风险设备: {len(low_risk_devices)} 个\n")
            f.write(f"- 无数据设备: {len(summary_df[summary_df['status'] == '无数据'])} 个\n")
            f.write(f"- 预测失败设备: {len(summary_df[summary_df['status'] == '预测失败'])} 个\n\n")
            
            if len(high_risk_devices) > 0:
                f.write("高风险设备详情:\n")
                f.write("-" * 60 + "\n")
                for _, device in high_risk_devices.iterrows():
                    f.write(f"设备ID: {device['device_id']}\n")
                    f.write(f"  - 最大故障概率: {device['max_fault_probability']:.2%}\n")
                    f.write(f"  - 平均故障概率: {device['avg_fault_probability']:.2%}\n")
                    f.write(f"  - 故障预测次数: {device['fault_predictions']}\n")
                    f.write(f"  - 极高风险记录数: {device['extreme_risk_count']}\n\n")
        
        print(f"\n报告已保存到 {report_path}")

def main():
    parser = argparse.ArgumentParser(description='设备故障批量预测')
    
    # 设备选择参数
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--device-ids', nargs='+', type=str,
                       help='指定要预测的设备ID列表')
    group.add_argument('--device-file', type=str,
                       help='包含设备ID的文件路径（txt或csv格式）')
    parser.add_argument('--max-devices', type=int, default=100,
                        help='未指定设备时，最多预测的设备数量 (默认: 100)')
    
    # 预测参数
    parser.add_argument('--model-path', type=str, default='fault_prediction_model.pkl',
                        help='模型文件路径 (默认: fault_prediction_model.pkl)')
    parser.add_argument('--hours', type=int, default=24,
                        help='获取最近多少小时的数据 (默认: 24)')
    
    # 风险阈值参数
    parser.add_argument('--high-risk-threshold', type=float, default=0.8,
                        help='高风险阈值 (默认: 0.8)')
    parser.add_argument('--medium-risk-threshold', type=float, default=0.5,
                        help='中风险阈值 (默认: 0.5)')
    
    # 输出参数
    parser.add_argument('--save-details', action='store_true', default=True,
                        help='是否保存详细预测结果 (默认: True)')
    parser.add_argument('--save-report', action='store_true', default=True,
                        help='是否保存文本报告 (默认: True)')
    
    args = parser.parse_args()
    
    # 执行批量预测
    batch_predict(args)

if __name__ == "__main__":
    main()
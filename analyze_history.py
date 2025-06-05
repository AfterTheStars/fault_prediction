#!/usr/bin/env python3
"""
设备故障历史数据分析脚本
用于分析历史预测数据和设备故障趋势
"""

import argparse
from fault_predictor_core import OptimizedDeviceFaultPredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# 数据库配置
DB_CONFIG = {
    'host': '117.72.55.146',
    'port': 1012,
    'database': 'postgres',
    'user': 'dbreader',
    'password': 'db@123456'
}

def analyze_device_history(predictor, device_id, days=30, save_plots=True):
    """
    分析单个设备的历史故障趋势
    """
    print(f"\n分析设备 {device_id} 最近 {days} 天的历史数据...")
    
    # 加载历史数据
    df = predictor.load_data(
        recent_days=days,
        device_id=device_id
    )
    
    if df is None or len(df) == 0:
        print(f"设备 {device_id} 无数据")
        return None
    
    # 进行预测
    predictions = predictor.predict(df)
    
    # 分析结果
    analysis = {
        'device_id': device_id,
        'total_records': len(predictions),
        'time_range': f"{predictions['f_time'].min()} 到 {predictions['f_time'].max()}",
        'fault_predictions': predictions['is_fault_predicted'].sum(),
        'fault_rate': predictions['is_fault_predicted'].mean(),
        'avg_fault_probability': predictions['fault_probability'].mean(),
        'max_fault_probability': predictions['fault_probability'].max(),
        'risk_distribution': predictions['risk_level'].value_counts().to_dict()
    }
    
    # 可视化分析
    if save_plots:
        plot_device_analysis(predictions, device_id, days)
    
    return analysis, predictions

def plot_device_analysis(predictions, device_id, days):
    """
    绘制设备分析图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'设备 {device_id} 最近 {days} 天故障分析', fontsize=16)
    
    # 1. 故障概率时间序列
    predictions_sorted = predictions.sort_values('f_time')
    axes[0, 0].plot(predictions_sorted['f_time'], 
                    predictions_sorted['fault_probability'], 
                    alpha=0.7, linewidth=1)
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='阈值=0.5')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('故障概率')
    axes[0, 0].set_title('故障概率时间序列')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 每日故障率
    predictions['date'] = predictions['f_time'].dt.date
    daily_fault_rate = predictions.groupby('date')['is_fault_predicted'].mean()
    axes[0, 1].bar(daily_fault_rate.index, daily_fault_rate.values)
    axes[0, 1].set_xlabel('日期')
    axes[0, 1].set_ylabel('故障率')
    axes[0, 1].set_title('每日故障率')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 传感器数据分布
    sensor_cols = ['f_amp', 'f_vol', 'f_temp', 'f_rate']
    for i, col in enumerate(sensor_cols):
        if col in predictions.columns:
            predictions[col].hist(bins=50, alpha=0.7, 
                                 label=col, ax=axes[1, 0])
    axes[1, 0].set_xlabel('传感器值')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('传感器数据分布')
    axes[1, 0].legend()
    
    # 4. 风险等级时间分布
    risk_pivot = predictions.pivot_table(
        index=predictions['f_time'].dt.date,
        columns='risk_level',
        values='f_device',
        aggfunc='count',
        fill_value=0
    )
    risk_pivot.plot(kind='area', stacked=True, ax=axes[1, 1],
                   color=['green', 'yellow', 'orange', 'red'])
    axes[1, 1].set_xlabel('日期')
    axes[1, 1].set_ylabel('记录数')
    axes[1, 1].set_title('风险等级时间分布')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    plot_dir = "analysis_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"device_{device_id}_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"分析图表已保存到 {plot_path}")

def compare_devices(predictor, device_ids, days=30):
    """
    比较多个设备的故障情况
    """
    print(f"\n比较 {len(device_ids)} 个设备最近 {days} 天的故障情况...")
    
    comparison_results = []
    all_predictions = []
    
    for device_id in device_ids:
        analysis, predictions = analyze_device_history(
            predictor, device_id, days, save_plots=False
        )
        
        if analysis:
            comparison_results.append(analysis)
            predictions['device_id'] = device_id
            all_predictions.append(predictions)
    
    if not comparison_results:
        print("没有设备数据可供比较")
        return None
    
    # 创建比较数据框
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values('avg_fault_probability', ascending=False)
    
    # 合并所有预测数据
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # 绘制比较图表
    plot_device_comparison(comparison_df, all_predictions_df, days)
    
    return comparison_df

def plot_device_comparison(comparison_df, all_predictions_df, days):
    """
    绘制设备比较图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'多设备故障比较分析（最近 {days} 天）', fontsize=16)
    
    # 1. 设备平均故障概率对比
    axes[0, 0].bar(range(len(comparison_df)), 
                   comparison_df['avg_fault_probability'])
    axes[0, 0].set_xlabel('设备')
    axes[0, 0].set_ylabel('平均故障概率')
    axes[0, 0].set_title('设备平均故障概率对比')
    axes[0, 0].set_xticks(range(len(comparison_df)))
    axes[0, 0].set_xticklabels(comparison_df['device_id'], rotation=45)
    
    # 2. 设备故障率对比
    axes[0, 1].bar(range(len(comparison_df)), 
                   comparison_df['fault_rate'])
    axes[0, 1].set_xlabel('设备')
    axes[0, 1].set_ylabel('故障率')
    axes[0, 1].set_title('设备故障率对比')
    axes[0, 1].set_xticks(range(len(comparison_df)))
    axes[0, 1].set_xticklabels(comparison_df['device_id'], rotation=45)
    
    # 3. 各设备故障概率箱线图
    all_predictions_df.boxplot(column='fault_probability', 
                               by='device_id', ax=axes[1, 0])
    axes[1, 0].set_xlabel('设备ID')
    axes[1, 0].set_ylabel('故障概率')
    axes[1, 0].set_title('各设备故障概率分布')
    
    # 4. 时间序列对比（选择前5个设备）
    top_devices = comparison_df.head(5)['device_id'].tolist()
    for device_id in top_devices:
        device_data = all_predictions_df[all_predictions_df['device_id'] == device_id]
        device_data_sorted = device_data.sort_values('f_time')
        
        # 按日聚合
        daily_avg = device_data_sorted.groupby(
            device_data_sorted['f_time'].dt.date
        )['fault_probability'].mean()
        
        axes[1, 1].plot(daily_avg.index, daily_avg.values, 
                       label=device_id, alpha=0.7)
    
    axes[1, 1].set_xlabel('日期')
    axes[1, 1].set_ylabel('日均故障概率')
    axes[1, 1].set_title('Top 5 高风险设备故障概率趋势')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    plot_dir = "analysis_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"device_comparison_{days}days.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"比较图表已保存到 {plot_path}")

def generate_analysis_report(predictor, device_ids, days=30):
    """
    生成综合分析报告
    """
    print("\n生成综合分析报告...")
    
    # 获取所有设备的比较数据
    comparison_df = compare_devices(predictor, device_ids, days)
    
    if comparison_df is None:
        print("无法生成报告")
        return
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"analysis_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("设备故障历史分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析时间范围: 最近 {days} 天\n")
        f.write(f"分析设备数: {len(device_ids)}\n")
        f.write("="*60 + "\n\n")
        
        # 概览统计
        f.write("概览统计:\n")
        f.write("-"*40 + "\n")
        f.write(f"总体平均故障概率: {comparison_df['avg_fault_probability'].mean():.2%}\n")
        f.write(f"总体平均故障率: {comparison_df['fault_rate'].mean():.2%}\n")
        f.write(f"最高风险设备: {comparison_df.iloc[0]['device_id']} "
                f"(故障概率: {comparison_df.iloc[0]['avg_fault_probability']:.2%})\n")
        f.write(f"最低风险设备: {comparison_df.iloc[-1]['device_id']} "
                f"(故障概率: {comparison_df.iloc[-1]['avg_fault_probability']:.2%})\n\n")
        
        # 高风险设备详情
        high_risk_devices = comparison_df[comparison_df['avg_fault_probability'] > 0.5]
        if len(high_risk_devices) > 0:
            f.write(f"高风险设备 (故障概率>50%): {len(high_risk_devices)} 个\n")
            f.write("-"*40 + "\n")
            for _, device in high_risk_devices.iterrows():
                f.write(f"\n设备ID: {device['device_id']}\n")
                f.write(f"  - 记录数: {device['total_records']}\n")
                f.write(f"  - 时间范围: {device['time_range']}\n")
                f.write(f"  - 平均故障概率: {device['avg_fault_probability']:.2%}\n")
                f.write(f"  - 最大故障概率: {device['max_fault_probability']:.2%}\n")
                f.write(f"  - 故障预测次数: {device['fault_predictions']}\n")
                f.write(f"  - 故障率: {device['fault_rate']:.2%}\n")
                
                # 风险分布
                if 'risk_distribution' in device and device['risk_distribution']:
                    f.write("  - 风险分布:\n")
                    for risk_level, count in device['risk_distribution'].items():
                        f.write(f"    * {risk_level}: {count} 次\n")
        
        # 设备排名列表
        f.write("\n\n所有设备风险排名:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'排名':<6} {'设备ID':<20} {'平均故障概率':<15} {'故障率':<10}\n")
        f.write("-"*40 + "\n")
        
        for i, (_, device) in enumerate(comparison_df.iterrows(), 1):
            f.write(f"{i:<6} {device['device_id']:<20} "
                    f"{device['avg_fault_probability']:<15.2%} "
                    f"{device['fault_rate']:<10.2%}\n")
    
    print(f"\n分析报告已保存到 {report_path}")
    
    # 同时保存详细数据
    csv_path = f"analysis_details_{timestamp}.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"详细数据已保存到 {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='设备故障历史数据分析')
    
    # 基本参数
    parser.add_argument('--model-path', type=str, default='fault_prediction_model.pkl',
                        help='模型文件路径 (默认: fault_prediction_model.pkl)')
    parser.add_argument('--days', type=int, default=30,
                        help='分析最近多少天的数据 (默认: 30)')
    
    # 分析模式
    subparsers = parser.add_subparsers(dest='mode', help='分析模式')
    
    # 单设备分析
    single_parser = subparsers.add_parser('single', help='分析单个设备')
    single_parser.add_argument('device_id', type=str, help='设备ID')
    
    # 多设备比较
    compare_parser = subparsers.add_parser('compare', help='比较多个设备')
    compare_parser.add_argument('--device-ids', nargs='+', type=str, required=True,
                               help='要比较的设备ID列表')
    
    # 综合报告
    report_parser = subparsers.add_parser('report', help='生成综合分析报告')
    report_parser.add_argument('--device-ids', nargs='+', type=str,
                              help='要分析的设备ID列表（不指定则分析所有设备）')
    report_parser.add_argument('--max-devices', type=int, default=50,
                              help='最多分析的设备数量 (默认: 50)')
    
    args = parser.parse_args()
    
    # 创建预测器并加载模型
    predictor = OptimizedDeviceFaultPredictor(DB_CONFIG)
    if not predictor.load_model(args.model_path):
        print(f"错误: 无法加载模型 {args.model_path}")
        return
    
    # 根据模式执行分析
    if args.mode == 'single':
        analysis, predictions = analyze_device_history(
            predictor, args.device_id, args.days
        )
        if analysis:
            print("\n分析结果:")
            for key, value in analysis.items():
                print(f"{key}: {value}")
    
    elif args.mode == 'compare':
        comparison_df = compare_devices(predictor, args.device_ids, args.days)
        if comparison_df is not None:
            print("\n设备比较结果:")
            print(comparison_df)
    
    elif args.mode == 'report':
        # 如果没有指定设备，获取活跃设备列表
        if args.device_ids:
            device_ids = args.device_ids
        else:
            print("获取活跃设备列表...")
            df_recent = predictor.load_data(recent_days=1, limit=10000)
            if df_recent is None or len(df_recent) == 0:
                print("错误: 无法获取设备列表")
                return
            device_ids = df_recent['f_device'].unique().tolist()[:args.max_devices]
            print(f"将分析 {len(device_ids)} 个设备")
        
        generate_analysis_report(predictor, device_ids, args.days)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
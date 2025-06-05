#!/usr/bin/env python3
"""
设备故障实时预测脚本
用于对指定设备进行实时故障预测
"""

import argparse
from fault_predictor_core import OptimizedDeviceFaultPredictor
import json
import sys

# 数据库配置
DB_CONFIG = {
    'host': '117.72.55.146',
    'port': 1012,
    'database': 'postgres',
    'user': 'dbreader',
    'password': 'db@123456'
}

def realtime_predict(args):
    """
    对指定设备进行实时预测
    """
    # 创建预测器实例
    predictor = OptimizedDeviceFaultPredictor(DB_CONFIG)
    
    # 加载模型
    if not predictor.load_model(args.model_path):
        print(f"错误: 无法加载模型文件 {args.model_path}")
        sys.exit(1)
    
    # 进行实时预测
    print(f"\n对设备 {args.device_id} 进行实时预测...")
    result = predictor.predict_realtime(
        device_id=args.device_id,
        recent_minutes=args.minutes
    )
    
    if result is None:
        print("错误: 预测失败")
        sys.exit(1)
    
    # 输出结果
    prediction_result = {
        'device_id': args.device_id,
        'timestamp': result['f_time'].strftime('%Y-%m-%d %H:%M:%S') if 'f_time' in result else None,
        'fault_probability': float(result['fault_probability']),
        'risk_level': str(result['risk_level']),
        'is_fault_predicted': bool(result['is_fault_predicted']),
        'prediction': '故障' if result['is_fault_predicted'] else '正常'
    }
    
    # 根据输出格式输出结果
    if args.output_format == 'json':
        print(json.dumps(prediction_result, indent=2, ensure_ascii=False))
    else:
        print(f"\n实时预测结果:")
        print(f"设备ID: {prediction_result['device_id']}")
        print(f"时间: {prediction_result['timestamp']}")
        print(f"故障概率: {prediction_result['fault_probability']:.2%}")
        print(f"风险等级: {prediction_result['risk_level']}")
        print(f"预测结果: {prediction_result['prediction']}")
    
    # 根据阈值返回退出码
    if result['fault_probability'] >= args.alert_threshold:
        sys.exit(2)  # 高风险退出码
    else:
        sys.exit(0)  # 正常退出码

def main():
    parser = argparse.ArgumentParser(description='设备故障实时预测')
    
    # 必需参数
    parser.add_argument('device_id', type=str,
                        help='要预测的设备ID')
    
    # 可选参数
    parser.add_argument('--model-path', type=str, default='fault_prediction_model.pkl',
                        help='模型文件路径 (默认: fault_prediction_model.pkl)')
    parser.add_argument('--minutes', type=int, default=10,
                        help='获取最近多少分钟的数据 (默认: 10)')
    parser.add_argument('--alert-threshold', type=float, default=0.8,
                        help='报警阈值，故障概率超过此值时返回非零退出码 (默认: 0.8)')
    parser.add_argument('--output-format', choices=['text', 'json'], default='text',
                        help='输出格式 (默认: text)')
    
    args = parser.parse_args()
    
    # 执行预测
    realtime_predict(args)

if __name__ == "__main__":
    main()
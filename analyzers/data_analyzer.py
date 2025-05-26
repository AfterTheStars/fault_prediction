import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from config.settings import SystemConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self):
        self.config = SystemConfig()
    
    def analyze_trends(self, df: pd.DataFrame) -> Dict:
        """分析设备数据趋势"""
        if df.empty:
            return {}
        
        # 确保时间列是datetime类型
        df['记录时间'] = pd.to_datetime(df['记录时间'])
        df = df.sort_values('记录时间')
        
        trends = {}
        
        # 分析各参数趋势
        for column in ['电流A', '电压V', '温度C', '负载率']:
            if column in df.columns and not df[column].isna().all():
                data = df[column].dropna()
                if len(data) > 0:
                    trends[column] = {
                        'current': float(data.iloc[-1]),
                        'avg': float(data.mean()),
                        'max': float(data.max()),
                        'min': float(data.min()),
                        'std': float(data.std()) if len(data) > 1 else 0,
                        'trend': self._calculate_trend(data),
                        'abnormal_rate': self._calculate_abnormal_rate(data, column)
                    }
        
        # 分析故障码分布
        fault_distribution = df['故障码'].value_counts().to_dict()
        
        # 分析工作状态分布
        work_status_distribution = df['工作信号'].value_counts().to_dict()
        
        return {
            'parameter_trends': trends,
            'fault_distribution': fault_distribution,
            'work_status_distribution': work_status_distribution,
            'data_points': len(df),
            'time_range': {
                'start': df['记录时间'].min().isoformat(),
                'end': df['记录时间'].max().isoformat()
            }
        }
    
    def _calculate_trend(self, data: pd.Series) -> float:
        """计算趋势（简单线性回归斜率）"""
        try:
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            y = data.values
            
            # 计算线性回归斜率
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        except:
            return 0.0
    
    def _calculate_abnormal_rate(self, data: pd.Series, parameter: str) -> float:
        """计算异常率"""
        try:
            normal_ranges = self.config.normal_ranges
            
            if parameter not in normal_ranges:
                return 0.0
            
            min_val, max_val = normal_ranges[parameter]
            abnormal_count = ((data < min_val) | (data > max_val)).sum()
            return float(abnormal_count / len(data) * 100) if len(data) > 0 else 0.0
        except:
            return 0.0
    
    def create_prediction_prompt(self, equipment_id: str, trends: Dict, 
                               fault_history: pd.DataFrame) -> str:
        """创建故障预测提示词"""
        
        prompt = f"""
请作为设备故障预测专家，分析设备 {equipment_id} 的故障风险：

=== 设备当前状态分析 ===
数据时间范围: {trends.get('time_range', {}).get('start', 'N/A')} 到 {trends.get('time_range', {}).get('end', 'N/A')}
数据点数量: {trends.get('data_points', 0)}

=== 关键参数趋势 ===
"""
        
        parameter_trends = trends.get('parameter_trends', {})
        for param, data in parameter_trends.items():
            status = "正常"
            if data['abnormal_rate'] > 20:
                status = "异常"
            elif data['abnormal_rate'] > 10:
                status = "注意"
            
            prompt += f"""
{param}:
- 当前值: {data['current']:.2f}
- 平均值: {data['avg']:.2f}
- 最大值: {data['max']:.2f}
- 最小值: {data['min']:.2f}
- 标准差: {data['std']:.2f}
- 变化趋势: {data['trend']:+.4f}
- 异常率: {data['abnormal_rate']:.1f}%
- 状态评估: {status}
"""
        
        # 故障码分析
        fault_dist = trends.get('fault_distribution', {})
        if fault_dist:
            prompt += "\n=== 故障码分布 ===\n"
            for fault_code, count in fault_dist.items():
                fault_desc = self.config.fault_code_mapping.get(fault_code, '未知故障')
                prompt += f"- {fault_code} ({fault_desc}): {count}次\n"
        
        # 工作状态分析
        work_dist = trends.get('work_status_distribution', {})
        if work_dist:
            prompt += "\n=== 工作状态分布 ===\n"
            for work_signal, count in work_dist.items():
                work_desc = self.config.work_signal_mapping.get(work_signal, '未知状态')
                prompt += f"- {work_signal} ({work_desc}): {count}次\n"
        
        # 历史故障分析
        if not fault_history.empty:
            prompt += f"\n=== 近期故障历史 ({len(fault_history)}条) ===\n"
            for _, row in fault_history.head(5).iterrows():
                fault_desc = self.config.fault_code_mapping.get(row['故障码'], '未知故障')
                prompt += f"- {row['记录时间']}: {row['故障码']} ({fault_desc})\n"
        
        prompt += """

    === 请提供以下分析 ===
    1. 设备健康状态评估 (健康/良好/注意/警告/危险)
    2. 故障风险预测 (未来24小时内故障概率 0-100%)
    3. 主要风险因素识别 (列出3个最重要的风险点)
    4. 故障类型预测 (最可能发生的故障类型)
    5. 维护建议 (具体的预防措施)
    6. 监控重点 (需要重点关注的参数)
    7. 预计故障时间 (如果有风险，预计多久内可能发生)

    请基于工业设备维护的专业知识和经验，提供准确、实用的分析结果。
    """
        
        return prompt
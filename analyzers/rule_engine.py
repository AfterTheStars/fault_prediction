from typing import Dict, List
from datetime import datetime
from config.settings import SystemConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class RuleEngine:
    """规则引擎"""
    
    def __init__(self):
        self.config = SystemConfig()
    
    def assess_risk(self, trends: Dict) -> Dict:
        """基于规则的风险评估"""
        risk_score = 0
        risk_factors = []
        alerts = []
        
        parameter_trends = trends.get('parameter_trends', {})
        
        # 电流异常检测
        if '电流A' in parameter_trends:
            current_data = parameter_trends['电流A']
            if current_data['current'] > 45 or current_data['current'] < 8:
                risk_score += 25
                risk_factors.append("电流异常")
                alerts.append(f"电流值 {current_data['current']:.1f}A 超出正常范围")
            
            if current_data['abnormal_rate'] > 15:
                risk_score += 15
                risk_factors.append("电流波动大")
        
        # 电压异常检测
        if '电压V' in parameter_trends:
            voltage_data = parameter_trends['电压V']
            if voltage_data['current'] > 400 or voltage_data['current'] < 360:
                risk_score += 20
                risk_factors.append("电压异常")
                alerts.append(f"电压值 {voltage_data['current']:.1f}V 超出正常范围")
        
        # 温度异常检测
        if '温度C' in parameter_trends:
            temp_data = parameter_trends['温度C']
            if temp_data['current'] > 75:
                risk_score += 30
                risk_factors.append("过温")
                alerts.append(f"温度 {temp_data['current']:.1f}°C 过高")
            elif temp_data['trend'] > 1.0:
                risk_score += 10
                risk_factors.append("温度上升趋势")
        
        # 负载率异常检测
        if '负载率' in parameter_trends:
            load_data = parameter_trends['负载率']
            if load_data['current'] > 90:
                risk_score += 25
                risk_factors.append("负载过高")
                alerts.append(f"负载率 {load_data['current']:.1f}% 过高")
            elif load_data['current'] < 10:
                risk_score += 15
                risk_factors.append("负载过低")
        
        # 故障频次检测
        fault_dist = trends.get('fault_distribution', {})
        if fault_dist:
            fault_count = sum(v for k, v in fault_dist.items() if k != '0')
            if fault_count > 10:
                risk_score += 20
                risk_factors.append("故障频发")
        
        # 确定风险等级
        if risk_score >= 70:
            risk_level = "危险"
        elif risk_score >= 50:
            risk_level = "警告"
        elif risk_score >= 30:
            risk_level = "注意"
        elif risk_score >= 10:
            risk_level = "良好"
        else:
            risk_level = "健康"
        
        return {
            "risk_level": risk_level,
            "risk_score": min(risk_score, 100),
            "fault_probability": min(risk_score * 1.2, 95),
            "risk_factors": risk_factors[:5],
            "alerts": alerts[:3],
            "assessment_time": datetime.now().isoformat()
        }
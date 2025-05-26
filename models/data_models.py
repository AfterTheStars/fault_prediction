from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class EquipmentData:
    """设备数据结构"""
    equipment_id: str
    fault_code: str
    work_signal: str
    record_time: datetime
    current_a: float
    voltage_v: float
    temperature_c: float
    load_rate_percent: float

@dataclass
class RiskAssessment:
    """风险评估结果"""
    risk_level: str  # 健康/良好/注意/警告/危险
    risk_score: float  # 0-100
    fault_probability: float  # 0-100
    risk_factors: List[str]
    alerts: List[str]
    assessment_time: datetime

@dataclass
class PredictionResult:
    """预测结果"""
    equipment_id: str
    prediction_timestamp: datetime
    data_analysis: Dict
    knowledge_retrieval: Dict
    ragflow_analysis: Dict
    llm_analysis: Dict
    rule_based_assessment: RiskAssessment
    error: Optional[str] = None
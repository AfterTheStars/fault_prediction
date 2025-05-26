from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = "117.72.55.146"
    port: int = 1012
    database: str = "postgres"
    user: str = "dbreader"
    password: str = "db@123456"
    
    # 表和字段映射
    table_name: str = "dj_data2"
    field_mapping = {
        'device': 'f_device',
        'fault_code': 'f_err_code',
        'work_signal': 'f_run_signal',
        'time': 'f_time',
        'current': 'f_amp',
        'voltage': 'f_vol',
        'temperature': 'f_temp',
        'load_rate': 'f_rate'
    }

@dataclass
class RAGflowConfig:
    """RAGflow配置"""
    base_url: str = "http://172.28.138.121"
    token: str = "ragflow-Y3NzUxZWE4MzdiNjExZjBhYWY5MDI0Mm"
    knowledge_base_id: str = "800ab7c4379811f0a4550242ac140007"
    timeout: int = 30

@dataclass
class LLMConfig:
    """大模型配置"""
    provider: str = "qwen"  # openai, claude, qwen, glm, etc.
    api_key: str = "sk-YH6u7ca5is1DzxRu6aubCw"
    base_url: str = "http://172.28.138.206:8000/v1"
    model: str = "Qwen3-30B-A3B"
    temperature: float = 0.3
    max_tokens: int = 2000

@dataclass
class SystemConfig:
    """系统配置"""
    log_file: str = "fault_prediction.log"
    alert_file: str = "alerts.log"
    
    # 正常参数范围
    normal_ranges = {
        '电流A': (5, 50),
        '电压V': (350, 410),
        '温度C': (10, 80),
        '负载率': (0, 100)
    }
    
    # 故障码映射
    fault_code_mapping = {
        '0': '正常',
        '0001': '过电流',
        '0002': '欠电压',
        '0003': '过电压',
        '0004': '过温',
        '0005': '负载异常',
        '0006': '通信故障',
        '0007': '传感器故障',
        '0008': '电机故障',
        '0009': '系统故障'
    }
    
    # 工作信号映射
    work_signal_mapping = {
        '1': '运行',
        '0': '停机',
        '2': '故障',
        '3': '维护',
        '4': '待机'
    }
import psycopg2
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
from config.settings import DatabaseConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.test_connection()
    
    def get_connection(self):
        """获取数据库连接"""
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password
        )
    
    def test_connection(self):
        """测试数据库连接"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            logger.info(f"数据库连接成功，版本: {version[0]}")
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def get_equipment_list(self) -> List[str]:
        """获取所有设备号列表"""
        try:
            conn = self.get_connection()
            query = f"""
                SELECT DISTINCT {self.config.field_mapping['device']} as 设备号 
                FROM {self.config.table_name}
                ORDER BY 设备号
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df['设备号'].tolist()
        except Exception as e:
            logger.error(f"获取设备列表失败: {str(e)}")
            return []
    
    def get_equipment_data(self, equipment_id: str, hours: int = 24) -> pd.DataFrame:
        """获取设备数据"""
        try:
            conn = self.get_connection()
            field_map = self.config.field_mapping
            
            query = f"""
                SELECT {field_map['device']} as 设备号,
                       {field_map['fault_code']} as 故障码,
                       {field_map['work_signal']} as 工作信号,
                       {field_map['time']} as 记录时间,
                       {field_map['current']} as 电流A,
                       {field_map['voltage']} as 电压V,
                       {field_map['temperature']} as 温度C,
                       {field_map['load_rate']} as 负载率
                FROM {self.config.table_name}
                WHERE {field_map['device']} = %s 
                  AND {field_map['time']} >= NOW() - INTERVAL '%s hours'
                ORDER BY {field_map['time']} DESC
            """
            df = pd.read_sql_query(query, conn, params=(equipment_id, hours))
            conn.close()
            return df
        except Exception as e:
            logger.error(f"获取设备数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_fault_history(self, equipment_id: str, days: int = 30) -> pd.DataFrame:
        """获取设备故障历史"""
        try:
            conn = self.get_connection()
            field_map = self.config.field_mapping
            
            query = f"""
                SELECT {field_map['device']} as 设备号,
                       {field_map['fault_code']} as 故障码,
                       {field_map['work_signal']} as 工作信号,
                       {field_map['time']} as 记录时间,
                       {field_map['current']} as 电流A,
                       {field_map['voltage']} as 电压V,
                       {field_map['temperature']} as 温度C,
                       {field_map['load_rate']} as 负载率
                FROM {self.config.table_name}
                WHERE {field_map['device']} = %s 
                  AND {field_map['fault_code']} != '0'
                  AND {field_map['time']} >= NOW() - INTERVAL '%s days'
                ORDER BY 记录时间 DESC
            """
            df = pd.read_sql_query(query, conn, params=(equipment_id, days))
            conn.close()
            return df
        except Exception as e:
            logger.error(f"获取故障历史失败: {str(e)}")
            return pd.DataFrame()
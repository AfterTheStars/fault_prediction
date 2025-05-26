from typing import List, Dict
from datetime import datetime
from config.settings import SystemConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.config = SystemConfig()
    
    def check_and_alert(self, results: List[Dict]):
        """检查高风险设备并发送告警"""
        high_risk_count = 0
        critical_equipment = []
        
        for result in results:
            if 'error' not in result:
                rule_assessment = result.get('rule_based_assessment', {})
                risk_level = rule_assessment.get('risk_level', '')
                
                if risk_level == '危险':
                    high_risk_count += 1
                    critical_equipment.append({
                        'equipment_id': result['equipment_id'],
                        'risk_score': rule_assessment.get('risk_score', 0),
                        'alerts': rule_assessment.get('alerts', [])
                    })
        
        if high_risk_count > 0:
            alert_message = f"发现 {high_risk_count} 台设备处于危险状态：\n"
            for equipment in critical_equipment:
                alert_message += f"- 设备 {equipment['equipment_id']}: 风险分数 {equipment['risk_score']}\n"
                for alert in equipment['alerts']:
                    alert_message += f"  * {alert}\n"
            
            logger.warning(alert_message)
            self._send_alert(alert_message)
    
    def _send_alert(self, message: str):
        """发送告警消息"""
        # 保存到告警日志
        alert_filename = f"alerts_{datetime.now().strftime('%Y%m%d')}.log"
        with open(alert_filename, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now()}] {message}\n\n")
        
        # TODO: 这里可以添加邮件发送、企业微信、钉钉等告警方式
        # self._send_email(message)
        # self._send_wechat(message)
        # self._send_dingtalk(message)
    
    def _send_email(self, message: str):
        """发送邮件告警（预留接口）"""
        pass
    
    def _send_wechat(self, message: str):
        """发送企业微信告警（预留接口）"""
        pass
    
    def _send_dingtalk(self, message: str):
        """发送钉钉告警（预留接口）"""
        pass
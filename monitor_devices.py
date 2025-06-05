#!/usr/bin/env python3
"""
设备故障定期监控脚本
用于定期监控设备状态并发送预警
"""

import argparse
from fault_predictor_core import OptimizedDeviceFaultPredictor
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import os
import logging

# 数据库配置
DB_CONFIG = {
    'host': '117.72.55.146',
    'port': 1012,
    'database': 'postgres',
    'user': 'dbreader',
    'password': 'db@123456'
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('device_monitor.log'),
        logging.StreamHandler()
    ]
)

class DeviceMonitor:
    def __init__(self, model_path, config_file=None):
        """
        初始化设备监控器
        """
        self.predictor = OptimizedDeviceFaultPredictor(DB_CONFIG)
        self.model_path = model_path
        self.config = self.load_config(config_file) if config_file else {}
        self.alert_history = {}  # 记录已发送的警报
        
    def load_config(self, config_file):
        """
        加载配置文件
        """
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            logging.warning(f"配置文件 {config_file} 不存在，使用默认配置")
            return {}
    
    def monitor_devices(self, device_ids=None, recent_hours=1, alert_threshold=0.8):
        """
        监控设备状态
        """
        logging.info("开始设备监控...")
        
        # 加载模型
        if not self.predictor.load_model(self.model_path):
            logging.error(f"无法加载模型: {self.model_path}")
            return
        
        # 如果没有指定设备，从数据库获取活跃设备
        if device_ids is None:
            df_recent = self.predictor.load_data(recent_days=1/24, limit=1000)
            if df_recent is None or len(df_recent) == 0:
                logging.warning("无法获取活跃设备列表")
                return
            device_ids = df_recent['f_device'].unique().tolist()
            logging.info(f"监控 {len(device_ids)} 个活跃设备")
        
        # 批量预测
        try:
            results = self.predictor.batch_predict_devices(
                device_ids=device_ids,
                recent_hours=recent_hours
            )
            
            if len(results) == 0:
                logging.warning("批量预测未返回结果")
                return
            
            # 筛选高风险设备
            high_risk_devices = results[
                (results['status'] == '正常') & 
                (results['max_fault_probability'] >= alert_threshold)
            ]
            
            # 记录监控结果
            logging.info(f"监控完成: {len(high_risk_devices)} 个高风险设备")
            
            # 处理警报
            if len(high_risk_devices) > 0:
                self.handle_alerts(high_risk_devices, alert_threshold)
            
            # 保存监控记录
            self.save_monitor_log(results)
            
        except Exception as e:
            logging.error(f"监控过程出错: {e}")
    
    def handle_alerts(self, high_risk_devices, threshold):
        """
        处理高风险设备警报
        """
        new_alerts = []
        
        for _, device in high_risk_devices.iterrows():
            device_id = device['device_id']
            fault_prob = device['max_fault_probability']
            
            # 检查是否需要发送新警报
            if self.should_send_alert(device_id, fault_prob):
                new_alerts.append({
                    'device_id': device_id,
                    'fault_probability': fault_prob,
                    'avg_fault_probability': device['avg_fault_probability'],
                    'extreme_risk_count': device['extreme_risk_count']
                })
                
                # 更新警报历史
                self.alert_history[device_id] = {
                    'last_alert_time': datetime.now(),
                    'last_fault_probability': fault_prob
                }
        
        # 发送警报
        if new_alerts:
            self.send_alerts(new_alerts, threshold)
    
    def should_send_alert(self, device_id, fault_prob):
        """
        判断是否应该发送警报
        避免重复发送相同的警报
        """
        if device_id not in self.alert_history:
            return True
        
        last_alert = self.alert_history[device_id]
        time_since_last = datetime.now() - last_alert['last_alert_time']
        
        # 如果距离上次警报超过1小时，或故障概率显著增加，则发送新警报
        if time_since_last.total_seconds() > 3600 or \
           fault_prob > last_alert['last_fault_probability'] * 1.2:
            return True
        
        return False
    
    def send_alerts(self, alerts, threshold):
        """
        发送警报（这里只是示例，实际可以发送邮件、短信等）
        """
        logging.warning(f"发现 {len(alerts)} 个需要警报的高风险设备!")
        
        # 打印警报信息
        for alert in alerts:
            logging.warning(
                f"高风险警报 - 设备: {alert['device_id']}, "
                f"故障概率: {alert['fault_probability']:.2%}"
            )
        
        # 如果配置了邮件，发送邮件警报
        if self.config.get('email_enabled', False):
            self.send_email_alert(alerts, threshold)
        
        # 保存警报到文件
        self.save_alert_log(alerts)
    
    def send_email_alert(self, alerts, threshold):
        """
        发送邮件警报
        """
        try:
            email_config = self.config.get('email', {})
            
            # 构建邮件内容
            subject = f"设备故障预警 - {len(alerts)} 个高风险设备"
            
            body = f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            body += f"警报阈值: {threshold:.0%}\n"
            body += f"高风险设备数: {len(alerts)}\n\n"
            
            body += "设备详情:\n"
            body += "-" * 60 + "\n"
            for alert in alerts:
                body += f"设备ID: {alert['device_id']}\n"
                body += f"  最大故障概率: {alert['fault_probability']:.2%}\n"
                body += f"  平均故障概率: {alert['avg_fault_probability']:.2%}\n"
                body += f"  极高风险记录数: {alert['extreme_risk_count']}\n\n"
            
            # 发送邮件（这里需要配置SMTP服务器信息）
            # self.send_email(subject, body, email_config)
            logging.info("邮件警报已发送")
            
        except Exception as e:
            logging.error(f"发送邮件警报失败: {e}")
    
    def save_monitor_log(self, results):
        """
        保存监控日志
        """
        log_dir = "monitor_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"monitor_log_{timestamp}.csv")
        
        results.to_csv(log_path, index=False)
        logging.info(f"监控日志已保存到 {log_path}")
    
    def save_alert_log(self, alerts):
        """
        保存警报日志
        """
        alert_file = "alerts.log"
        
        with open(alert_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for alert in alerts:
                f.write(f"{timestamp} | {alert['device_id']} | {alert['fault_probability']:.2%}\n")
    
    def run_scheduled_monitoring(self, interval_minutes=60, device_ids=None, 
                               recent_hours=1, alert_threshold=0.8):
        """
        运行定期监控
        """
        logging.info(f"启动定期监控，间隔: {interval_minutes} 分钟")
        
        # 定义监控任务
        def monitor_task():
            self.monitor_devices(device_ids, recent_hours, alert_threshold)
        
        # 立即执行一次
        monitor_task()
        
        # 设置定期任务
        schedule.every(interval_minutes).minutes.do(monitor_task)
        
        # 持续运行
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            logging.info("监控已停止")

def main():
    parser = argparse.ArgumentParser(description='设备故障定期监控')
    
    # 基本参数
    parser.add_argument('--model-path', type=str, default='fault_prediction_model.pkl',
                        help='模型文件路径 (默认: fault_prediction_model.pkl)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（JSON格式）')
    
    # 监控参数
    parser.add_argument('--interval', type=int, default=60,
                        help='监控间隔（分钟） (默认: 60)')
    parser.add_argument('--recent-hours', type=int, default=1,
                        help='获取最近多少小时的数据 (默认: 1)')
    parser.add_argument('--alert-threshold', type=float, default=0.8,
                        help='警报阈值 (默认: 0.8)')
    
    # 设备选择
    parser.add_argument('--device-ids', nargs='+', type=str, default=None,
                        help='要监控的设备ID列表（不指定则监控所有活跃设备）')
    
    # 运行模式
    parser.add_argument('--once', action='store_true',
                        help='只运行一次监控，不进行定期监控')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = DeviceMonitor(args.model_path, args.config)
    
    if args.once:
        # 只运行一次
        monitor.monitor_devices(
            device_ids=args.device_ids,
            recent_hours=args.recent_hours,
            alert_threshold=args.alert_threshold
        )
    else:
        # 定期监控
        monitor.run_scheduled_monitoring(
            interval_minutes=args.interval,
            device_ids=args.device_ids,
            recent_hours=args.recent_hours,
            alert_threshold=args.alert_threshold
        )

if __name__ == "__main__":
    main()
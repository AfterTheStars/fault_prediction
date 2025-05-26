import schedule
import time
import threading
from typing import List, Optional
from datetime import datetime
from services.prediction_service import PredictionService
from reports.report_generator import ReportGenerator
from utils.alerts import AlertManager
from utils.logger import get_logger

logger = get_logger(__name__)

class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, prediction_service: PredictionService):
        self.prediction_service = prediction_service
        self.report_generator = ReportGenerator()
        self.alert_manager = AlertManager()
        self.running = False
        self.thread = None
    
    def scheduled_prediction(self, equipment_ids: Optional[List[str]] = None, 
                           interval_hours: int = 6):
        """定时预测任务"""
        def job():
            logger.info("开始执行定时故障预测任务")
            try:
                # 执行批量预测
                results = self.prediction_service.batch_predict(equipment_ids)
                
                # 保存结果
                self.report_generator.save_results(results)
                
                # 生成报告
                report = self.report_generator.generate_summary_report(results)
                
                # 保存报告
                report_filename = f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                logger.info(f"定时预测任务完成，报告已保存到: {report_filename}")
                
                # 检查高风险设备并发送告警
                self.alert_manager.check_and_alert(results)
                
            except Exception as e:
                logger.error(f"定时预测任务失败: {str(e)}")
        
        # 设置定时任务
        schedule.every(interval_hours).hours.do(job)
        
        logger.info(f"定时预测任务已设置，每{interval_hours}小时执行一次")
    
    def start(self):
        """启动调度器"""
        if self.running:
            logger.warning("调度器已在运行")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("任务调度器已启动")
    
    def stop(self):
        """停止调度器"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("任务调度器已停止")
    
    def _run(self):
        """运行调度器"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
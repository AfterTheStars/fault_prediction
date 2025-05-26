#!/usr/bin/env python3
"""
设备故障预测系统主程序
"""
import argparse
from config.settings import DatabaseConfig, RAGflowConfig, LLMConfig
from database.db_manager import DatabaseManager
from services.ragflow_service import RAGflowService
from services.llm_service import LLMService
from services.prediction_service import PredictionService
from reports.report_generator import ReportGenerator
from reports.dashboard import DashboardManager
from tasks.scheduler import TaskScheduler
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='设备故障预测系统')
    parser.add_argument('--mode', choices=['single', 'batch', 'scheduled', 'dashboard'], 
                       default='batch', help='运行模式')
    parser.add_argument('--equipment', type=str, help='设备ID（单设备预测模式）')
    parser.add_argument('--interval', type=int, default=6, help='定时任务间隔（小时）')
    parser.add_argument('--export', choices=['excel', 'csv'], help='导出报告格式')
    
    args = parser.parse_args()
    
    # 初始化配置
    db_config = DatabaseConfig()
    ragflow_config = RAGflowConfig()
    llm_config = LLMConfig()
    
    # 初始化服务
    logger.info("初始化系统服务...")
    db_manager = DatabaseManager(db_config)
    ragflow_service = RAGflowService(ragflow_config)
    llm_service = LLMService(llm_config)
    prediction_service = PredictionService(db_manager, ragflow_service, llm_service)
    report_generator = ReportGenerator()
    
    try:
        if args.mode == 'single':
            # 单设备预测模式
            if not args.equipment:
                equipment_list = db_manager.get_equipment_list()
                if equipment_list:
                    args.equipment = equipment_list[0]
                    logger.info(f"未指定设备，使用第一个设备: {args.equipment}")
                else:
                    logger.error("无可用设备")
                    return
            
            logger.info(f"预测设备: {args.equipment}")
            result = prediction_service.predict_equipment_fault(args.equipment)
            
            risk_level = result.get('rule_based_assessment', {}).get('risk_level', '未知')
            logger.info(f"预测完成，风险等级: {risk_level}")
            
            # 保存结果
            report_generator.save_results([result])
            
        elif args.mode == 'batch':
            # 批量预测模式
            logger.info("开始批量故障预测...")
            equipment_list = db_manager.get_equipment_list()
            
            # 限制预测数量（演示用）
            sample_size = min(5, len(equipment_list))
            results = prediction_service.batch_predict(equipment_list[:sample_size])
            
            # 生成报告
            summary_report = report_generator.generate_summary_report(results)
            print(summary_report)
            
            # 保存结果
            report_generator.save_results(results)
            
            # 导出报告
            if args.export:
                export_file = report_generator.export_to_excel(results, args.export)
                if export_file:
                    logger.info(f"报告已导出: {export_file}")
            
        elif args.mode == 'scheduled':
            # 定时任务模式
            logger.info("启动定时预测任务...")
            scheduler = TaskScheduler(prediction_service)
            scheduler.scheduled_prediction(interval_hours=args.interval)
            scheduler.start()
            
            # 保持程序运行
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                scheduler.stop()
                logger.info("定时任务已停止")
                
        elif args.mode == 'dashboard':
            # 仪表板模式
            logger.info("获取仪表板数据...")
            dashboard_manager = DashboardManager(db_manager)
            dashboard_data = dashboard_manager.get_health_dashboard()
            
            print("\n=== 设备健康仪表板 ===")
            print(f"设备总数: {dashboard_data.get('total_equipment', 0)}")
            print(f"更新时间: {dashboard_data.get('last_update', '')}")
            print(f"\n风险分布: {dashboard_data.get('risk_distribution', {})}")
            print(f"\n故障趋势: {dashboard_data.get('fault_trends', {})}")
            print(f"\n参数汇总: {dashboard_data.get('parameter_summary', {})}")
            
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行错误: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
import pandas as pd
import json
from typing import Dict, List
from datetime import datetime
from config.settings import SystemConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.config = SystemConfig()
    
    def generate_summary_report(self, prediction_results: List[Dict]) -> str:
        """生成汇总报告"""
        report = f"""
=== 设备故障预测汇总报告 ===
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
预测设备总数: {len(prediction_results)}

=== 风险等级统计 ===
"""
        
        # 统计风险等级
        risk_stats = {}
        high_risk_equipment = []
        
        for result in prediction_results:
            if 'error' not in result:
                rule_assessment = result.get('rule_based_assessment', {})
                risk_level = rule_assessment.get('risk_level', '未知')
                risk_score = rule_assessment.get('risk_score', 0)
                
                risk_stats[risk_level] = risk_stats.get(risk_level, 0) + 1
                
                if risk_level in ['危险', '警告']:
                    high_risk_equipment.append({
                        'equipment_id': result['equipment_id'],
                        'risk_level': risk_level,
                        'risk_score': risk_score
                    })
        
        for level, count in risk_stats.items():
            report += f"- {level}: {count}台\n"
        
        # 高风险设备列表
        if high_risk_equipment:
            report += f"\n=== 高风险设备 ({len(high_risk_equipment)}台) ===\n"
            for equipment in sorted(high_risk_equipment, key=lambda x: x['risk_score'], reverse=True):
                report += f"- {equipment['equipment_id']}: {equipment['risk_level']} (风险分数: {equipment['risk_score']})\n"
        
        # 统计故障类型
        fault_types = {}
        for result in prediction_results:
            if 'error' not in result:
                fault_dist = result.get('data_analysis', {}).get('trends', {}).get('fault_distribution', {})
                for fault_code, count in fault_dist.items():
                    if fault_code != '0000':
                        fault_desc = self.config.fault_code_mapping.get(fault_code, '未知故障')
                        fault_types[fault_desc] = fault_types.get(fault_desc, 0) + count
        
        if fault_types:
            report += f"\n=== 故障类型分布 ===\n"
            for fault_type, count in sorted(fault_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                report += f"- {fault_type}: {count}次\n"
        
        # 生成建议
        report += f"\n=== 维护建议 ===\n"
        if len(high_risk_equipment) > 0:
            report += f"1. 立即检查 {len(high_risk_equipment)} 台高风险设备\n"
        report += f"2. 加强日常巡检，重点关注温度和电流参数\n"
        report += f"3. 建立预防性维护计划\n"
        report += f"4. 及时更换老化部件\n"
        
        return report
    
    def save_results(self, results: List[Dict], filename: str = None):
        """保存预测结果到文件"""
        if filename is None:
            filename = f"fault_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"预测结果已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
    
    def export_to_excel(self, results: List[Dict], format_type: str = 'excel') -> str:
        """导出预测报告到Excel"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format_type.lower() == 'excel':
                filename = f"fault_prediction_report_{timestamp}.xlsx"
                
                # 准备数据
                report_data = []
                for result in results:
                    if 'error' not in result:
                        equipment_id = result['equipment_id']
                        rule_assessment = result.get('rule_based_assessment', {})
                        data_analysis = result.get('data_analysis', {})
                        trends = data_analysis.get('trends', {})
                        parameter_trends = trends.get('parameter_trends', {})
                        
                        row = {
                            '设备号': equipment_id,
                            '预测时间': result.get('prediction_timestamp', ''),
                            '风险等级': rule_assessment.get('risk_level', ''),
                            '风险分数': rule_assessment.get('risk_score', 0),
                            '故障概率(%)': rule_assessment.get('fault_probability', 0),
                            '主要风险因素': ', '.join(rule_assessment.get('risk_factors', [])),
                            '告警信息': ', '.join(rule_assessment.get('alerts', [])),
                            '当前故障码': data_analysis.get('latest_fault_code', ''),
                            '工作信号': data_analysis.get('current_work_signal', ''),
                            '历史故障次数': data_analysis.get('fault_history_count', 0)
                        }
                        
                        # 添加参数信息
                        for param in ['电流A', '电压V', '温度C', '负载率']:
                            if param in parameter_trends:
                                row[f'{param}_当前值'] = parameter_trends[param].get('current', 0)
                                row[f'{param}_异常率'] = parameter_trends[param].get('abnormal_rate', 0)
                        
                        report_data.append(row)
                
                # 导出Excel
                df = pd.DataFrame(report_data)
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='故障预测报告', index=False)
                    
                    # 添加汇总页
                    summary_data = []
                    risk_stats = df['风险等级'].value_counts().to_dict()
                    for level, count in risk_stats.items():
                        summary_data.append({'风险等级': level, '设备数量': count})
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='汇总统计', index=False)
                
                logger.info(f"Excel报告已导出: {filename}")
                return filename
                
            else:  # CSV格式
                filename = f"fault_prediction_report_{timestamp}.csv"
                
                report_data = []
                for result in results:
                    if 'error' not in result:
                        equipment_id = result['equipment_id']
                        rule_assessment = result.get('rule_based_assessment', {})
                        
                        report_data.append({
                            '设备号': equipment_id,
                            '风险等级': rule_assessment.get('risk_level', ''),
                            '风险分数': rule_assessment.get('risk_score', 0),
                            '故障概率': rule_assessment.get('fault_probability', 0),
                            '风险因素': ', '.join(rule_assessment.get('risk_factors', [])),
                            '预测时间': result.get('prediction_timestamp', '')
                        })
                
                df = pd.DataFrame(report_data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                
                logger.info(f"CSV报告已导出: {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"导出报告失败: {str(e)}")
            return ""
from typing import Dict, List, Optional
from datetime import datetime
import time
from database.db_manager import DatabaseManager
from services.ragflow_service import RAGflowService
from services.llm_service import LLMService
from analyzers.data_analyzer import DataAnalyzer
from analyzers.rule_engine import RuleEngine
from models.data_models import PredictionResult
from utils.logger import get_logger

logger = get_logger(__name__)

class PredictionService:
    """预测服务"""
    
    def __init__(self, db_manager: DatabaseManager, 
                 ragflow_service: RAGflowService,
                 llm_service: Optional[LLMService] = None):
        self.db_manager = db_manager
        self.ragflow_service = ragflow_service
        self.llm_service = llm_service
        self.data_analyzer = DataAnalyzer()
        self.rule_engine = RuleEngine()
    
    def predict_equipment_fault(self, equipment_id: str) -> Dict:
        """预测单个设备故障"""
        try:
            logger.info(f"开始预测设备 {equipment_id} 的故障风险")
            
            # 1. 获取设备数据
            latest_data = self.db_manager.get_equipment_data(equipment_id, hours=24)
            if latest_data.empty:
                return {
                    "equipment_id": equipment_id,
                    "error": "无法获取设备数据",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 2. 分析数据趋势
            trends = self.data_analyzer.analyze_trends(latest_data)
            
            # 3. 获取故障历史
            fault_history = self.db_manager.get_fault_history(equipment_id, days=30)
            
            # 4. 创建预测提示
            prompt = self.data_analyzer.create_prediction_prompt(
                equipment_id, trends, fault_history
            )
            
            # 5. 查询知识库
            knowledge_result = self.ragflow_service.query_knowledge(
                f"设备故障预测 {equipment_id} 电流异常 温度过高 负载异常"
            )
            
            # 6. RAGflow对话分析
            chat_result = self.ragflow_service.chat(prompt)
            
            # 7. LLM深度分析
            llm_analysis = {}
            if self.llm_service:
                context = ""
                if knowledge_result and 'chunks' in knowledge_result:
                    context = "相关技术资料摘要：\n"
                    for chunk in knowledge_result['chunks'][:3]:
                        context += f"- {chunk.get('content', '')[:200]}...\n"
                
                llm_analysis = self.llm_service.analyze(prompt, context)
            
            # 8. 规则引擎评估
            rule_based_risk = self.rule_engine.assess_risk(trends)
            
            # 9. 整合预测结果
            prediction_result = {
                "equipment_id": equipment_id,
                "prediction_timestamp": datetime.now().isoformat(),
                "data_analysis": {
                    "trends": trends,
                    "fault_history_count": len(fault_history),
                    "latest_fault_code": latest_data.iloc[0]['故障码'] if not latest_data.empty else None,
                    "current_work_signal": latest_data.iloc[0]['工作信号'] if not latest_data.empty else None
                },
                "knowledge_retrieval": knowledge_result,
                "ragflow_analysis": chat_result,
                "llm_analysis": llm_analysis,
                "rule_based_assessment": rule_based_risk,
                "raw_prompt": prompt
            }
            
            logger.info(f"设备 {equipment_id} 故障预测完成")
            return prediction_result
            
        except Exception as e:
            logger.error(f"设备 {equipment_id} 故障预测失败: {str(e)}")
            return {
                "equipment_id": equipment_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_predict(self, equipment_ids: Optional[List[str]] = None) -> List[Dict]:
        """批量预测设备故障"""
        if equipment_ids is None:
            equipment_ids = self.db_manager.get_equipment_list()
        
        results = []
        for equipment_id in equipment_ids:
            try:
                result = self.predict_equipment_fault(equipment_id)
                results.append(result)
                time.sleep(1)  # 避免API调用过频
            except Exception as e:
                logger.error(f"批量预测设备 {equipment_id} 失败: {str(e)}")
                results.append({
                    "equipment_id": equipment_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
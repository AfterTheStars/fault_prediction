import requests
from typing import Dict, Optional
from datetime import datetime
from config.settings import RAGflowConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class RAGflowService:
    """RAGflow服务"""
    
    def __init__(self, config: RAGflowConfig):
        self.config = config
        self.headers = {
            'Authorization': f'Bearer {config.token}',
            'Content-Type': 'application/json'
        }
    
    def query_knowledge(self, query: str) -> Dict:
        """查询知识库"""
        try:
            search_url = f"{self.config.base_url}/api/v1/retrieval"
            payload = {
                "knowledge_base_id": self.config.knowledge_base_id,
                "question": query,
                "similarity_threshold": 0.1,
                "vector_similarity_weight": 0.3,
                "top_k": 5
            }
            
            response = requests.post(
                search_url, 
                json=payload, 
                headers=self.headers, 
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(result)
                logger.info("RAGflow知识检索成功")
                return result
            else:
                logger.error(f"RAGflow查询失败: {response.status_code}, {response.text}")
                return {"error": f"查询失败: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"RAGflow查询异常: {str(e)}")
            return {"error": str(e)}
    
    def chat(self, question: str, session_id: Optional[str] = None) -> Dict:
        """对话分析"""
        try:
            chat_url = f"{self.config.base_url}/api/v1/chat/{self.config.knowledge_base_id}"
            payload = {
                "question": question,
                "stream": False,
                "session_id": session_id or f"equipment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            response = requests.post(
                chat_url, 
                json=payload, 
                headers=self.headers, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("RAGflow对话分析成功")
                return result
            else:
                logger.error(f"RAGflow对话失败: {response.status_code}, {response.text}")
                return {"error": f"对话失败: {response.status_code}"}
        
        except Exception as e:
            logger.error(f"RAGflow对话异常: {str(e)}")
            return {"error": str(e)}
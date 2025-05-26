from typing import Dict, Optional
from config.settings import LLMConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class LLMService:
    """大模型服务"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化大模型客户端"""
        try:
            if self.config.provider.lower() in ["openai", "qwen", "glm"]:
                import openai
                self.client = openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
                logger.info(f"{self.config.provider}客户端初始化成功")
                
            elif self.config.provider.lower() == "claude":
                import anthropic
                self.client = anthropic.Anthropic(
                    api_key=self.config.api_key
                )
                logger.info("Claude客户端初始化成功")
                
            else:
                logger.warning(f"不支持的LLM提供商: {self.config.provider}")
                
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {str(e)}")
            self.client = None
    
    def analyze(self, prompt: str, context: str = "") -> Dict:
        """使用大模型进行分析"""
        try:
            if not self.client:
                return {"error": "LLM客户端未初始化"}
            
            # 构建完整的分析提示
            full_prompt = f"""
你是一位资深的工业设备故障预测专家，具有20年的设备维护和故障诊断经验。

{context}

{prompt}

请基于你的专业知识和经验，提供准确、实用的分析结果。分析应该包含：
1. 专业的技术判断
2. 具体的数据解读
3. 可执行的建议措施
4. 风险等级评估

请用专业、简洁的语言回答。
"""
            
            if self.config.provider.lower() == "claude":
                # Claude API调用
                response = self.client.messages.create(
                    model=self.config.model or "claude-3-sonnet-20240229",
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )
                analysis_result = response.content[0].text
                
            else:
                # OpenAI兼容API调用
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "你是一位资深的工业设备故障预测专家。"},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                analysis_result = response.choices[0].message.content
            
            logger.info("LLM分析完成")
            return {
                "analysis": analysis_result,
                "model": self.config.model,
                "provider": self.config.provider
            }
            
        except Exception as e:
            logger.error(f"LLM分析失败: {str(e)}")
            return {"error": str(e)}
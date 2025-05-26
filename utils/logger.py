import logging
from datetime import datetime
from config.settings import SystemConfig

def get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    config = SystemConfig()
    
    logger = logging.getLogger(name)
    
    # 如果logger已经有处理器，则不重复添加
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(
        config.log_file, 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
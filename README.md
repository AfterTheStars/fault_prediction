# 设备故障预测系统

基于PostgreSQL + RAGflow + LLM的智能设备故障预测系统。

## 功能特性

- **数据采集与分析**：从PostgreSQL数据库实时获取设备运行数据
- **智能预测**：结合规则引擎、知识库和大模型进行故障预测
- **多模式运行**：支持单设备预测、批量预测、定时任务和仪表板
- **风险评估**：五级风险等级评估（健康/良好/注意/警告/危险）
- **报告生成**：自动生成预测报告，支持Excel和CSV导出
- **告警通知**：高风险设备自动告警

## 系统架构


## 安装部署

### 1. 环境要求

- Python 3.8+
- PostgreSQL 数据库
- RAGflow 服务
- 大模型服务（支持OpenAI、Claude、通义千问、智谱GLM等）

### 2. 安装依赖

```bash
pip install -r requirements.txt


###编辑 config/settings.py，配置数据库、RAGflow和大模型服务：

# 数据库配置
@dataclass
class DatabaseConfig:
    host: str = "your_db_host"
    port: int = 5432
    database: str = "your_database"
    user: str = "your_user"
    password: str = "your_password"

# RAGflow配置
@dataclass
class RAGflowConfig:
    base_url: str = "http://your_ragflow_url"
    token: str = "your_ragflow_token"
    knowledge_base_id: str = "your_knowledge_base_id"

# 大模型配置
@dataclass
class LLMConfig:
    provider: str = "qwen"  # 或 openai, claude, glm
    api_key: str = "your_api_key"
    base_url: str = "your_api_url"
    model: str = "your_model_name"


    使用方法
1. 单设备预测
python main.py --mode single --equipment DEVICE001

2. 批量预测
python main.py --mode batch --export excel
3. 定时任务
python main.py --mode dashboard
4. 查看仪表板
python main.py --mode dashboard



运行模式说明
Single Mode（单设备模式）

对指定设备进行故障预测
输出详细的分析结果
适用于定点排查

Batch Mode（批量模式）

对多台设备进行批量预测
生成汇总报告
支持导出Excel/CSV

Scheduled Mode（定时模式）

按设定间隔自动执行预测
自动生成报告和告警
适用于持续监控

Dashboard Mode（仪表板模式）

显示设备健康状态概览
统计风险分布和故障趋势
适用于整体监控

风险等级说明
风险等级风险分数说明健康0-10设备运行正常，无需特殊关注良好10-30设备基本正常，建议常规维护注意30-50存在潜在风险，需要加强监控警告50-70风险较高，建议尽快检查维护危险70-100高风险状态，需要立即处理
主要功能模块
1. 数据分析器 (DataAnalyzer)

参数趋势分析
异常率计算
故障分布统计

2. 规则引擎 (RuleEngine)

基于阈值的风险评估
多维度综合评分
告警规则匹配

3. RAGflow服务 (RAGflowService)

知识库检索
智能对话分析
历史案例匹配

4. 大模型服务 (LLMService)

深度故障分析
维护建议生成
风险预测推理

输出文件说明

fault_prediction_YYYYMMDD_HHMMSS.json: 预测结果详细数据
fault_prediction_report_YYYYMMDD_HHMMSS.xlsx: Excel格式报告
prediction_report_YYYYMMDD_HHMMSS.txt: 文本格式汇总报告
alerts_YYYYMMDD.log: 告警日志
fault_prediction.log: 系统运行日志

扩展开发
添加新的告警方式
在 utils/alerts.py 中实现新的告警方法：
pythondef _send_email(self, message: str):
    """发送邮件告警"""
    # 实现邮件发送逻辑
    pass

def _send_wechat(self, message: str):
    """发送企业微信告警"""
    # 实现企业微信发送逻辑
    pass
添加机器学习模型
在 models/ml_models.py 中实现ML预测：
pythondef train(self, X, y):
    """训练故障预测模型"""
    self.model = RandomForestClassifier()
    self.model.fit(X, y)

def predict(self, features: Dict) -> Dict:
    """使用ML模型预测故障"""
    # 实现预测逻辑
    pass
自定义风险规则
在 analyzers/rule_engine.py 中添加新规则：
python# 添加新的检测规则
if '振动值' in parameter_trends:
    vibration_data = parameter_trends['振动值']
    if vibration_data['current'] > threshold:
        risk_score += 20
        risk_factors.append("振动异常")
故障排除
1. 数据库连接失败

检查数据库配置是否正确
确认网络连接正常
验证用户权限

2. RAGflow服务异常

检查RAGflow服务是否启动
验证token是否有效
确认知识库ID正确

3. 大模型调用失败

检查API密钥是否正确
确认API地址可访问
验证模型名称是否支持

性能优化建议

批量预测优化

调整批次大小
使用多线程处理
优化数据库查询


内存使用优化

限制数据加载量
及时释放大对象
使用生成器处理大数据


API调用优化

实现请求缓存
合理设置超时
添加重试机制



许可证
MIT License
作者
Your Name
更新日志
v1.0.0 (2024-01-01)

初始版本发布
实现基础故障预测功能
支持多种运行模式


## 26. 项目部署脚本 setup.sh
```bash
#!/bin/bash
# 设备故障预测系统部署脚本

echo "=== 设备故障预测系统部署 ==="

# 创建项目目录结构
echo "创建目录结构..."
mkdir -p fault_prediction_system/{config,database,models,services,analyzers,reports,utils,tasks}

# 创建__init__.py文件
echo "创建__init__.py文件..."
touch fault_prediction_system/config/__init__.py
touch fault_prediction_system/database/__init__.py
touch fault_prediction_system/models/__init__.py
touch fault_prediction_system/services/__init__.py
touch fault_prediction_system/analyzers/__init__.py
touch fault_prediction_system/reports/__init__.py
touch fault_prediction_system/utils/__init__.py
touch fault_prediction_system/tasks/__init__.py

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 创建日志目录
echo "创建日志目录..."
mkdir -p logs

# 设置执行权限
echo "设置执行权限..."
chmod +x main.py

echo "部署完成！"
echo "请修改 config/settings.py 中的配置信息后运行："
echo "python main.py --help"
使用说明

创建项目结构：按照上述目录结构创建文件夹
复制代码：将各个模块的代码复制到对应的文件中
配置修改：根据实际情况修改 config/settings.py 中的数据库、RAGflow和大模型配置
安装依赖：运行 pip install -r requirements.txt
运行系统：使用 python main.py 启动系统

这个重构后的代码具有以下优点：

模块化设计：各功能模块独立，便于维护和扩展
配置集中管理：所有配置集中在 config/settings.py
清晰的分层架构：数据层、服务层、业务逻辑层分离
易于扩展：可以方便地添加新的服务、分析器或告警方式
完善的日志记录：统一的日志管理
灵活的运行模式：支持多种运行方式满足不同需求


创建项目目录结构
将对应代码复制到各个文件中
修改配置文件中的数据库和服务连接信息
安装依赖：pip install -r requirements.txt
运行系统：

单设备预测：python main.py --mode single --equipment DEVICE001
批量预测：python main.py --mode batch
定时任务：python main.py --mode scheduled
仪表板：python main.py --mode dashboard
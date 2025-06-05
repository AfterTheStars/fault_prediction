# 设备故障预测系统

基于XGBoost的工业设备故障预测系统，用于实时监控设备状态并预测潜在故障。

## 功能特性

- **故障预测**: 使用XGBoost算法预测设备故障概率
- **实时监控**: 支持单设备实时预测和批量设备监控
- **历史分析**: 分析设备历史数据，识别故障趋势
- **自动告警**: 定期监控并自动发送高风险设备警报
- **可视化报告**: 生成图表和分析报告

## 系统架构

```
fault_predictor_core.py    # 核心预测器类
├── train_model.py         # 模型训练脚本
├── realtime_predict.py    # 实时预测脚本
├── batch_predict.py       # 批量预测脚本
├── monitor_devices.py     # 定期监控脚本
└── analyze_history.py     # 历史分析脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用指南

### 1. 训练模型

首次使用需要训练模型：

```bash
# 使用最近7天数据训练
python train_model.py --days 7

# 指定数据量和设备
python train_model.py --days 30 --limit 500000 --device-id DEVICE001

# 不使用数据重采样
python train_model.py --no-resampling
```

### 2. 实时预测

对单个设备进行实时故障预测：

```bash
# 基本用法
python realtime_predict.py DEVICE001

# 指定时间范围和输出格式
python realtime_predict.py DEVICE001 --minutes 30 --output-format json

# 设置告警阈值
python realtime_predict.py DEVICE001 --alert-threshold 0.7
```

### 3. 批量预测

批量预测多个设备：

```bash
# 预测指定设备
python batch_predict.py --device-ids DEVICE001 DEVICE002 DEVICE003

# 从文件加载设备列表
python batch_predict.py --device-file devices.txt

# 预测所有活跃设备
python batch_predict.py --max-devices 100
```

### 4. 定期监控

启动定期监控服务：

```bash
# 每小时监控一次
python monitor_devices.py --interval 60

# 监控指定设备，每30分钟一次
python monitor_devices.py --device-ids DEVICE001 DEVICE002 --interval 30

# 只运行一次监控
python monitor_devices.py --once
```

### 5. 历史分析

分析设备历史数据：

```bash
# 分析单个设备
python analyze_history.py single DEVICE001 --days 30

# 比较多个设备
python analyze_history.py compare --device-ids DEVICE001 DEVICE002 DEVICE003

# 生成综合报告
python analyze_history.py report --days 7
```

## 配置说明

### 数据库配置

在各脚本中修改 `DB_CONFIG`：

```python
DB_CONFIG = {
    'host': 'your_host',
    'port': 5432,
    'database': 'your_database',
    'user': 'your_user',
    'password': 'your_password'
}
```

### 监控配置文件

创建 `monitor_config.json`：

```json
{
    "email_enabled": true,
    "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender": "your_email@gmail.com",
        "password": "your_password",
        "recipients": ["alert@example.com"]
    },
    "alert_rules": {
        "high_risk_threshold": 0.8,
        "medium_risk_threshold": 0.5
    }
}
```

## 输出文件

- **模型文件**: `fault_prediction_model_YYYYMMDD_HHMMSS.pkl`
- **预测结果**: `fault_predictions.csv`
- **批量预测报告**: `batch_prediction_report_YYYYMMDD_HHMMSS.txt`
- **监控日志**: `monitor_logs/monitor_log_YYYYMMDD_HHMMSS.csv`
- **分析图表**: `analysis_plots/device_DEVICEID_analysis.png`
- **分析报告**: `analysis_report_YYYYMMDD_HHMMSS.txt`

## 故障判定逻辑

系统通过以下方式判定故障：

1. **错误码**: `f_err_code != '0'`
2. **报警信号**: `f_alarm == 1`
3. **参数异常**:
   - 温度过高/过低 (2%/98%分位数)
   - 电流过高 (98%分位数)
   - 转速过低 (2%分位数)
   - 参数急剧变化 (95%分位数)

## 风险等级

- **低风险**: 故障概率 < 30%
- **中风险**: 30% ≤ 故障概率 < 60%
- **高风险**: 60% ≤ 故障概率 < 80%
- **极高风险**: 故障概率 ≥ 80%

## 注意事项

1. **数据要求**: 训练模型至少需要5个故障样本
2. **内存使用**: 大量数据训练时注意内存使用，建议使用 `--limit` 参数
3. **模型更新**: 建议定期重新训练模型以适应新的故障模式
4. **告警频率**: 避免告警过于频繁，系统会自动过滤1小时内的重复告警

## 性能优化建议

1. **训练优化**:
   - 使用 `--limit` 限制数据量
   - 关闭交叉验证 `--no-validate`
   - 减少特征工程复杂度

2. **预测优化**:
   - 批量预测比单个预测更高效
   - 适当增加监控间隔减少数据库负载
   - 使用缓存避免重复计算

3. **存储优化**:
   - 定期清理历史日志文件
   - 压缩存档旧的分析报告
   - 只保留最新的几个模型文件

## 故障排查

### 常见问题

1. **数据库连接失败**
   - 检查网络连接
   - 验证数据库配置信息
   - 确认数据库服务正常运行

2. **模型训练失败**
   - 检查数据中是否有足够的故障样本
   - 验证数据格式是否正确
   - 查看是否有缺失的必要字段

3. **预测结果异常**
   - 确认使用了正确的模型文件
   - 检查输入数据的时间范围
   - 验证特征工程是否正确执行

### 日志文件

- **系统日志**: `device_monitor.log`
- **告警日志**: `alerts.log`
- **错误追踪**: 查看各脚本的控制台输出

## 扩展开发

### 添加新特征

在 `fault_predictor_core.py` 的 `enhanced_feature_engineering` 方法中添加：

```python
# 示例：添加功率效率特征
df['power_efficiency'] = df['f_rate'] / (df['f_amp'] * df['f_vol'] + 1e-8)
```

### 自定义告警规则

在 `monitor_devices.py` 中修改 `should_send_alert` 方法：

```python
def should_send_alert(self, device_id, fault_prob):
    # 自定义告警逻辑
    if fault_prob > 0.9:  # 极高风险立即告警
        return True
    # 其他条件...
```

### 集成外部系统

可以通过以下方式集成：

1. **REST API**: 将预测功能封装为API服务
2. **消息队列**: 使用RabbitMQ/Kafka发送告警
3. **数据库集成**: 将预测结果写入数据库
4. **监控系统**: 集成到Prometheus/Grafana

## 版本历史

- v1.0.0: 初始版本，支持XGBoost模型训练和预测
- 后续版本规划：
  - 支持更多机器学习算法
  - 添加Web界面
  - 支持实时流数据处理
  - 增加更多可视化功能

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请联系项目维护者。
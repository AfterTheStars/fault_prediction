import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class OptimizedDeviceFaultPredictor:
    def __init__(self, db_config):
        """
        初始化优化的故障预测器
        """
        self.db_config = db_config
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None
        self.feature_cols = None
        self.model_trained = False
        
    def connect_db(self):
        """
        连接PostgreSQL数据库
        """
        try:
            print("尝试连接数据库...")
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            
            engine = create_engine('postgresql://', creator=lambda: conn)
            
            print("数据库连接成功")
            return engine
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return None
    
    def load_data(self, start_time=None, end_time=None, recent_days=None, limit=None, device_id=None):
        """
        从数据库加载数据，优化版本
        """
        engine = self.connect_db()
        if engine is None:
            return None
        
        # 构建基础查询语句
        query = """
        SELECT 
            f_device,
            f_err_code,
            f_run_signal,
            f_time,
            f_amp,
            f_vol,
            f_temp,
            f_rate,
            f_alarm,
            f_name
        FROM dj_data2_copy1
        WHERE f_amp IS NOT NULL 
          AND f_vol IS NOT NULL 
          AND f_temp IS NOT NULL 
          AND f_rate IS NOT NULL
        """
        
        # 处理设备ID条件
        if device_id:
            if isinstance(device_id, list):
                device_list = "','".join(map(str, device_id))
                query += f" AND f_device IN ('{device_list}')"
            else:
                query += f" AND f_device = '{device_id}'"
        
        # 处理时间条件
        time_conditions = []
        
        if recent_days is not None:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=recent_days)
            time_conditions.append(f"f_time >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'")
            print(f"加载最近 {recent_days} 天的数据")
        else:
            if start_time is not None:
                if isinstance(start_time, str):
                    start_time = pd.to_datetime(start_time)
                time_conditions.append(f"f_time >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'")
                
            if end_time is not None:
                if isinstance(end_time, str):
                    end_time = pd.to_datetime(end_time)
                time_conditions.append(f"f_time <= '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'")
        
        if time_conditions:
            query += " AND " + " AND ".join(time_conditions)
        
        query += " ORDER BY f_time DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            print(f"执行查询...")
            df = pd.read_sql(query, engine)
            print(f"成功加载 {len(df)} 条数据")
            
            if len(df) > 0:
                df['f_time'] = pd.to_datetime(df['f_time'])
                print(f"数据时间范围: {df['f_time'].min()} 到 {df['f_time'].max()}")
                print(f"设备数量: {df['f_device'].nunique()} 个")
            
            return df
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def enhanced_feature_engineering(self, df, for_prediction=False):
        """
        增强的特征工程
        for_prediction: 是否用于预测（预测时不创建故障标签）
        """
        df['f_time'] = pd.to_datetime(df['f_time'])
        df = df.sort_values(['f_device', 'f_time'])
        
        # 基础时间特征
        df['hour'] = df['f_time'].dt.hour
        df['day_of_week'] = df['f_time'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # 设备运行时间特征
        df['time_since_start'] = df.groupby('f_device')['f_time'].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 3600
        )
        
        # 核心传感器特征（减少计算量）
        for col in ['f_amp', 'f_vol', 'f_temp', 'f_rate']:
            # 短期移动平均 (5个点)
            df[f'{col}_ma5'] = df.groupby('f_device')[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            # 短期标准差
            df[f'{col}_std5'] = df.groupby('f_device')[col].transform(
                lambda x: x.rolling(5, min_periods=1).std()
            )
            # 变化率
            df[f'{col}_change'] = df.groupby('f_device')[col].transform(
                lambda x: x.diff()
            )
            # 与设备平均值的偏差
            df[f'{col}_dev'] = df.groupby('f_device')[col].transform(
                lambda x: x - x.mean()
            )
        
        # 组合特征
        df['power'] = df['f_amp'] * df['f_vol']
        df['temp_amp_ratio'] = df['f_temp'] / (df['f_amp'] + 1e-8)
        df['efficiency'] = df['f_rate'] / (df['power'] + 1e-8)
        
        # 只在训练时创建故障标签
        if not for_prediction:
            # 增强的故障标签创建
            # 方法1: 基于错误码
            fault_from_error = (df['f_err_code'] != '0') & (df['f_err_code'].notna())
            
            # 方法2: 基于报警信号
            fault_from_alarm = (df['f_alarm'] == 1) | (df['f_alarm'] == '1')
            
            # 方法3: 基于参数异常（更宽松的条件）
            # 使用更极端的阈值来增加故障样本
            temp_high = df['f_temp'] > df['f_temp'].quantile(0.98)
            temp_low = df['f_temp'] < df['f_temp'].quantile(0.02)
            amp_high = df['f_amp'] > df['f_amp'].quantile(0.98)
            rate_low = df['f_rate'] < df['f_rate'].quantile(0.02)
            
            # 基于参数变化异常
            temp_change_extreme = abs(df['f_temp_change']) > df['f_temp_change'].abs().quantile(0.95)
            amp_change_extreme = abs(df['f_amp_change']) > df['f_amp_change'].abs().quantile(0.95)
            
            fault_from_params = temp_high | temp_low | amp_high | rate_low | temp_change_extreme | amp_change_extreme
            
            # 综合故障标签
            df['is_fault'] = (fault_from_error | fault_from_alarm | fault_from_params).astype(int)
            
            # 统计各种故障来源
            print(f"\n故障来源统计:")
            print(f"- 错误码故障: {fault_from_error.sum()}")
            print(f"- 报警故障: {fault_from_alarm.sum()}")
            print(f"- 参数异常故障: {fault_from_params.sum()}")
            print(f"- 总故障数: {df['is_fault'].sum()}")
            print(f"- 故障率: {df['is_fault'].mean():.4f}")
        
        # 填充缺失值
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def prepare_features(self, df):
        """
        准备特征，减少特征数量以提高训练速度
        """
        # 选择重要特征
        feature_cols = [
            # 原始传感器数据
            'f_run_signal', 'f_amp', 'f_vol', 'f_temp', 'f_rate',
            # 时间特征
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            # 设备运行特征
            'time_since_start',
            # 核心统计特征
            'f_amp_ma5', 'f_vol_ma5', 'f_temp_ma5', 'f_rate_ma5',
            'f_amp_std5', 'f_vol_std5', 'f_temp_std5', 'f_rate_std5',
            'f_amp_change', 'f_vol_change', 'f_temp_change', 'f_rate_change',
            'f_amp_dev', 'f_vol_dev', 'f_temp_dev', 'f_rate_dev',
            # 组合特征
            'power', 'temp_amp_ratio', 'efficiency'
        ]
        
        # 保存特征列名
        self.feature_cols = feature_cols
        
        X = df[feature_cols]
        
        # 如果有故障标签，返回X和y；否则只返回X
        if 'is_fault' in df.columns:
            y = df['is_fault']
            return X, y, feature_cols
        else:
            return X, None, feature_cols
    
    def handle_imbalanced_data(self, X, y, method='smote'):
        """
        处理不平衡数据
        """
        print(f"\n原始数据分布:")
        print(f"正常样本: {(y==0).sum()}")
        print(f"故障样本: {(y==1).sum()}")
        
        if method == 'smote':
            # 使用SMOTE过采样
            smote = SMOTE(random_state=42, k_neighbors=min(5, (y==1).sum()-1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            # 不处理
            X_resampled, y_resampled = X, y
        
        print(f"\n重采样后数据分布:")
        print(f"正常样本: {(y_resampled==0).sum()}")
        print(f"故障样本: {(y_resampled==1).sum()}")
        
        return X_resampled, y_resampled
    
    def train_xgboost_model(self, X, y, use_resampling=True):
        """
        训练XGBoost模型
        """
        # 首先检查数据分布
        unique_classes = y.unique()
        class_counts = y.value_counts()
        
        print(f"\n类别分布:")
        print(class_counts)
        
        if len(unique_classes) < 2:
            print("数据中只有一个类别，无法进行训练")
            return None
        
        # 处理不平衡数据
        if use_resampling and class_counts.min() >= 5:
            X_balanced, y_balanced = self.handle_imbalanced_data(X, y, method='smote')
        else:
            X_balanced, y_balanced = X, y
            print("跳过重采样，使用原始数据")
        
        # 划分数据集
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42, 
                stratify=y_balanced if len(y_balanced.unique()) > 1 else None
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42
            )
        
        print("\n开始训练XGBoost模型...")
        
        # XGBoost模型配置
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = 0
        
        print(f"\nXGBoost - AUC Score: {auc_score:.4f}")
        print(f"分类报告:\n{classification_report(y_test, y_pred)}")
        
        # 特征重要性分析
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n特征重要性 Top 10:")
        print(self.feature_importance.head(10))
        
        self.model_trained = True
        self.model_use_scaled = False  # XGBoost不需要标准化
        
        return self.model
    
    def predict(self, df, threshold=0.5):
        """
        对新数据进行预测
        
        参数:
        df: 待预测的数据框
        threshold: 故障概率阈值（默认0.5）
        
        返回:
        包含预测结果的数据框
        """
        if not self.model_trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        # 特征工程
        print("进行特征工程...")
        df_features = self.enhanced_feature_engineering(df, for_prediction=True)
        
        # 准备特征
        X, _, _ = self.prepare_features(df_features)
        
        # 预测
        print("进行预测...")
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # 创建结果数据框
        result_df = df.copy()
        result_df['fault_prediction'] = predictions
        result_df['fault_probability'] = probabilities
        result_df['is_fault_predicted'] = (probabilities >= threshold).astype(int)
        
        # 添加故障风险等级
        result_df['risk_level'] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['低风险', '中风险', '高风险', '极高风险']
        )
        
        print(f"\n预测完成:")
        print(f"- 预测样本数: {len(result_df)}")
        print(f"- 预测故障数: {result_df['is_fault_predicted'].sum()}")
        print(f"- 预测故障率: {result_df['is_fault_predicted'].mean():.4f}")
        
        # 按风险等级统计
        print("\n风险等级分布:")
        print(result_df['risk_level'].value_counts().sort_index())
        
        return result_df
    
    def predict_realtime(self, device_id, recent_minutes=10):
        """
        实时预测指定设备的故障风险
        
        参数:
        device_id: 设备ID
        recent_minutes: 获取最近多少分钟的数据
        
        返回:
        预测结果
        """
        # 获取最新数据
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=recent_minutes)
        
        print(f"\n获取设备 {device_id} 最近 {recent_minutes} 分钟的数据...")
        df = self.load_data(
            start_time=start_time,
            end_time=end_time,
            device_id=device_id
        )
        
        if df is None or len(df) == 0:
            print("未获取到数据")
            return None
        
        # 进行预测
        predictions = self.predict(df)
        
        # 获取最新预测结果
        latest_prediction = predictions.iloc[0]
        
        print(f"\n设备 {device_id} 实时预测结果:")
        print(f"- 故障概率: {latest_prediction['fault_probability']:.2%}")
        print(f"- 风险等级: {latest_prediction['risk_level']}")
        print(f"- 预测结果: {'故障' if latest_prediction['is_fault_predicted'] else '正常'}")
        
        return latest_prediction
    
    def batch_predict_devices(self, device_ids, recent_hours=24):
        """
        批量预测多个设备的故障风险
        
        参数:
        device_ids: 设备ID列表
        recent_hours: 获取最近多少小时的数据
        
        返回:
        各设备的预测结果汇总
        """
        results = []
        
        for device_id in device_ids:
            print(f"\n处理设备 {device_id}...")
            
            # 获取数据
            df = self.load_data(
                recent_days=recent_hours/24,
                device_id=device_id
            )
            
            if df is None or len(df) == 0:
                print(f"设备 {device_id} 无数据")
                # 为无数据的设备添加默认记录
                device_summary = {
                    'device_id': device_id,
                    'total_records': 0,
                    'fault_predictions': 0,
                    'avg_fault_probability': 0,
                    'max_fault_probability': 0,
                    'high_risk_count': 0,
                    'extreme_risk_count': 0,
                    'status': '无数据'
                }
                results.append(device_summary)
                continue
            
            # 预测
            try:
                predictions = self.predict(df)
                
                # 汇总结果
                device_summary = {
                    'device_id': device_id,
                    'total_records': len(predictions),
                    'fault_predictions': predictions['is_fault_predicted'].sum(),
                    'avg_fault_probability': predictions['fault_probability'].mean(),
                    'max_fault_probability': predictions['fault_probability'].max(),
                    'high_risk_count': (predictions['risk_level'] == '高风险').sum(),
                    'extreme_risk_count': (predictions['risk_level'] == '极高风险').sum(),
                    'status': '正常'
                }
                
                results.append(device_summary)
            except Exception as e:
                print(f"设备 {device_id} 预测失败: {e}")
                # 为预测失败的设备添加默认记录
                device_summary = {
                    'device_id': device_id,
                    'total_records': len(df),
                    'fault_predictions': 0,
                    'avg_fault_probability': 0,
                    'max_fault_probability': 0,
                    'high_risk_count': 0,
                    'extreme_risk_count': 0,
                    'status': '预测失败'
                }
                results.append(device_summary)
        
        # 检查是否有结果
        if not results:
            print("\n没有任何设备数据可以分析")
            return pd.DataFrame()  # 返回空DataFrame
        
        # 创建汇总数据框
        summary_df = pd.DataFrame(results)
        
        # 只有在DataFrame不为空且包含所需列时才排序
        if len(summary_df) > 0 and 'max_fault_probability' in summary_df.columns:
            summary_df = summary_df.sort_values('max_fault_probability', ascending=False)
        
        print("\n批量预测结果汇总:")
        print(summary_df)
        
        # 统计信息
        total_devices = len(device_ids)
        devices_with_data = len(summary_df[summary_df['status'] == '正常'])
        devices_no_data = len(summary_df[summary_df['status'] == '无数据'])
        devices_failed = len(summary_df[summary_df['status'] == '预测失败'])
        
        print(f"\n统计信息:")
        print(f"- 总设备数: {total_devices}")
        print(f"- 有数据设备: {devices_with_data}")
        print(f"- 无数据设备: {devices_no_data}")
        print(f"- 预测失败设备: {devices_failed}")
        
        return summary_df
    
    def save_model(self, model_path='fault_prediction_model.pkl'):
        """
        保存模型
        """
        if not self.model_trained:
            print("模型尚未训练")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance,
            'model_use_scaled': getattr(self, 'model_use_scaled', False)
        }
        
        joblib.dump(model_data, model_path)
        print(f"模型已保存到 {model_path}")
    
    def load_model(self, model_path='fault_prediction_model.pkl'):
        """
        加载模型
        """
        if not os.path.exists(model_path):
            print(f"模型文件 {model_path} 不存在")
            return False
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.feature_importance = model_data['feature_importance']
        self.model_use_scaled = model_data.get('model_use_scaled', False)
        self.model_trained = True
        
        print(f"模型已从 {model_path} 加载")
        return True
    
    def plot_prediction_results(self, predictions, save_path=None):
        """
        可视化预测结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 故障概率分布
        axes[0, 0].hist(predictions['fault_probability'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('故障概率')
        axes[0, 0].set_ylabel('数量')
        axes[0, 0].set_title('故障概率分布')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='阈值=0.5')
        axes[0, 0].legend()
        
        # 2. 风险等级分布
        risk_counts = predictions['risk_level'].value_counts()
        axes[0, 1].bar(risk_counts.index, risk_counts.values, 
                       color=['green', 'yellow', 'orange', 'red'])
        axes[0, 1].set_xlabel('风险等级')
        axes[0, 1].set_ylabel('数量')
        axes[0, 1].set_title('风险等级分布')
        
        # 3. 时间序列故障概率
        if 'f_time' in predictions.columns:
            predictions_sorted = predictions.sort_values('f_time')
            axes[1, 0].plot(predictions_sorted['f_time'], 
                           predictions_sorted['fault_probability'], 
                           alpha=0.7)
            axes[1, 0].set_xlabel('时间')
            axes[1, 0].set_ylabel('故障概率')
            axes[1, 0].set_title('故障概率时间序列')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 设备故障率对比（如果有多个设备）
        if 'f_device' in predictions.columns and predictions['f_device'].nunique() > 1:
            device_fault_rates = predictions.groupby('f_device')['is_fault_predicted'].mean()
            device_fault_rates = device_fault_rates.sort_values(ascending=False).head(10)
            axes[1, 1].bar(range(len(device_fault_rates)), device_fault_rates.values)
            axes[1, 1].set_xlabel('设备')
            axes[1, 1].set_ylabel('故障率')
            axes[1, 1].set_title('Top 10 高故障率设备')
            axes[1, 1].set_xticks(range(len(device_fault_rates)))
            axes[1, 1].set_xticklabels(device_fault_rates.index, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存到 {save_path}")
        
        plt.show()
    
    def validate_model_robustness(self, X, y, n_splits=5):
        """
        交叉验证评估模型稳定性
        """
        if self.model is None:
            print("模型尚未训练")
            return
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_scores = []
        
        print(f"\n进行 {n_splits} 折交叉验证...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # 训练模型副本
            model_copy = xgb.XGBClassifier(**self.model.get_params())
            model_copy.fit(X_train_fold, y_train_fold)
            y_pred_proba = model_copy.predict_proba(X_val_fold)[:, 1]
            
            try:
                auc = roc_auc_score(y_val_fold, y_pred_proba)
                auc_scores.append(auc)
                print(f"Fold {fold+1}: AUC = {auc:.4f}")
            except:
                print(f"Fold {fold+1}: AUC计算失败")
        
        if auc_scores:
            print(f"\n交叉验证结果:")
            print(f"平均AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
            print(f"AUC范围: {min(auc_scores):.4f} - {max(auc_scores):.4f}")
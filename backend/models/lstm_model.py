"""LSTM预测模型"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from loguru import logger


class LSTMModel(nn.Module):
    """LSTM模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMPredictor:
    """LSTM预测器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.hidden_size = self.config.get('hidden_size', 64)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.seq_length = self.config.get('seq_length', 20)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'close', 
                    feature_cols: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c.startswith('alpha_')]
        
        if not feature_cols:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # 提取特征和目标
        features = df[feature_cols].values
        target = df[target_col].pct_change().shift(-1).values  # 预测下一日收益
        
        # 移除NaN
        valid_idx = ~np.isnan(target)
        features = features[valid_idx]
        target = target[valid_idx]
        
        # 标准化
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        
        # 创建序列
        X, y = [], []
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i+self.seq_length])
            y.append(target[i+self.seq_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_size: int):
        """构建模型"""
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        logger.info(f"LSTM模型构建完成: input_size={input_size}, hidden_size={self.hidden_size}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """训练模型"""
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        # 转换为tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            # Mini-batch训练
            for i in range(0, len(X_train_t), self.batch_size):
                batch_X = X_train_t[i:i+self.batch_size]
                batch_y = y_train_t[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / (len(X_train_t) // self.batch_size)
            train_losses.append(avg_train_loss)
            
            # 验证
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t).squeeze()
                    val_loss = criterion(val_outputs, y_val_t).item()
                    val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_t).squeeze().cpu().numpy()
        
        return predictions
    
    def predict_signal(self, df: pd.DataFrame, feature_cols: list = None) -> Dict:
        """预测交易信号"""
        X, _ = self.prepare_data(df, feature_cols=feature_cols)
        
        if len(X) == 0:
            return {'signal': '持有', 'confidence': 0}
        
        # 使用最后一个序列预测
        last_sequence = X[-1:]
        prediction = self.predict(last_sequence)[0]
        
        # 转换为信号
        if prediction > 0.02:
            signal = '买入'
            confidence = min(abs(prediction) * 10, 1.0)
        elif prediction < -0.02:
            signal = '卖出'
            confidence = min(abs(prediction) * 10, 1.0)
        else:
            signal = '持有'
            confidence = 1 - abs(prediction) * 10
        
        return {
            'signal': signal,
            'confidence': confidence,
            'prediction': prediction
        }
    
    def save_model(self, path: str):
        """保存模型"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'scaler': self.scaler
            }, path)
            logger.info(f"模型已保存: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.scaler = checkpoint['scaler']
        
        # 重建模型
        input_size = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[1]
        self.build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已加载: {path}")

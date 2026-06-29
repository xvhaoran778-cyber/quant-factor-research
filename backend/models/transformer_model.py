"""Transformer预测模型"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from typing import Dict, Tuple
from loguru import logger


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    """Transformer模型"""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 取最后一个时间步
        out = x[:, -1, :]
        out = self.fc(out)
        return out


class TransformerPredictor:
    """Transformer预测器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.d_model = self.config.get('d_model', 64)
        self.nhead = self.config.get('nhead', 4)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.learning_rate = self.config.get('learning_rate', 0.0001)
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.seq_length = self.config.get('seq_length', 30)
        
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
        
        features = df[feature_cols].values
        target = df[target_col].pct_change().shift(-1).values
        
        valid_idx = ~np.isnan(target)
        features = features[valid_idx]
        target = target[valid_idx]
        
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i+self.seq_length])
            y.append(target[i+self.seq_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_size: int):
        """构建模型"""
        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        logger.info(f"Transformer模型构建完成: input_size={input_size}, d_model={self.d_model}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """训练模型"""
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
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
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
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
        
        last_sequence = X[-1:]
        prediction = self.predict(last_sequence)[0]
        
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
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.scaler = checkpoint['scaler']
        
        input_size = checkpoint['model_state_dict']['input_projection.weight'].shape[1]
        self.build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])

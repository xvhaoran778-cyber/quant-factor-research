"""强化学习交易模型"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from loguru import logger
from collections import deque
import random


class TradingEnvironment:
    """交易环境"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: list = None,
                 initial_capital: float = 1000000, commission: float = 0.0003):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols or ['open', 'high', 'low', 'close', 'volume']
        self.initial_capital = initial_capital
        self.commission = commission
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0  # 持仓数量
        self.portfolio_value = self.initial_capital
        self.done = False
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        if self.current_step >= len(self.df):
            return np.zeros(len(self.feature_cols) + 2)
        
        # 特征
        features = self.df.iloc[self.current_step][self.feature_cols].values
        
        # 持仓状态
        position_ratio = self.position * self.df.iloc[self.current_step]['close'] / self.portfolio_value if self.portfolio_value > 0 else 0
        capital_ratio = self.capital / self.portfolio_value if self.portfolio_value > 0 else 1
        
        state = np.concatenate([features, [position_ratio, capital_ratio]])
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作
        
        Args:
            action: 0=持有, 1=买入, 2=卖出
        
        Returns:
            state, reward, done, info
        """
        if self.done:
            return self._get_state(), 0, True, {}
        
        current_price = self.df.iloc[self.current_step]['close']
        
        # 执行交易
        if action == 1:  # 买入
            if self.position == 0 and self.capital > current_price * 100:
                # 买入100股
                buy_amount = current_price * 100
                commission = buy_amount * self.commission
                self.capital -= (buy_amount + commission)
                self.position = 100
        elif action == 2:  # 卖出
            if self.position > 0:
                # 卖出全部
                sell_amount = current_price * self.position
                commission = sell_amount * self.commission
                self.capital += (sell_amount - commission)
                self.position = 0
        
        # 移动到下一步
        self.current_step += 1
        
        if self.current_step >= len(self.df):
            self.done = True
            # 强制平仓
            if self.position > 0:
                last_price = self.df.iloc[-1]['close']
                self.capital += last_price * self.position
                self.position = 0
        
        # 计算新的投资组合价值
        if not self.done:
            new_price = self.df.iloc[self.current_step]['close']
            self.portfolio_value = self.capital + self.position * new_price
        else:
            self.portfolio_value = self.capital
        
        # 计算奖励
        reward = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # 如果持仓，考虑价格变化
        if not self.done and self.position > 0:
            price_change = (self.df.iloc[self.current_step]['close'] / current_price - 1)
            reward += price_change * 10  # 放大持仓收益
        
        info = {
            'portfolio_value': self.portfolio_value,
            'capital': self.capital,
            'position': self.position
        }
        
        return self._get_state(), reward, self.done, info


class DQNetwork(nn.Module):
    """DQN网络"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class RLTrader:
    """强化学习交易器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.hidden_size = self.config.get('hidden_size', 64)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.batch_size = self.config.get('batch_size', 64)
        self.buffer_size = self.config.get('buffer_size', 10000)
        self.target_update = self.config.get('target_update', 10)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon = self.epsilon_start
        
        self.env = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.memory = ReplayBuffer(self.buffer_size)
    
    def build_model(self, state_size: int, action_size: int = 3):
        """构建模型"""
        self.policy_net = DQNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        logger.info(f"DQN模型构建完成: state_size={state_size}, action_size={action_size}")
    
    def train(self, df: pd.DataFrame, feature_cols: list = None, 
              episodes: int = 100) -> Dict:
        """训练模型"""
        # 创建环境
        self.env = TradingEnvironment(df, feature_cols)
        
        # 构建模型
        state_size = len(feature_cols or ['open', 'high', 'low', 'close', 'volume']) + 2
        if self.policy_net is None:
            self.build_model(state_size)
        
        # 训练
        episode_rewards = []
        episode_values = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while not self.env.done:
                # 选择动作
                action = self._select_action(state)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.memory.push(state, action, reward, next_state, done)
                
                # 训练
                if len(self.memory) >= self.batch_size:
                    self._train_step()
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # 更新epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # 更新目标网络
            if (episode + 1) % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            episode_rewards.append(total_reward)
            episode_values.append(self.env.portfolio_value)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_value = np.mean(episode_values[-10:])
                logger.info(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.4f}, Avg Value: {avg_value:.2f}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_values': episode_values
        }
    
    def _select_action(self, state: np.ndarray) -> int:
        """选择动作（epsilon-greedy）"""
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()
    
    def _train_step(self):
        """训练一步"""
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def predict_signal(self, df: pd.DataFrame, feature_cols: list = None) -> Dict:
        """预测交易信号"""
        if self.policy_net is None:
            return {'signal': '持有', 'confidence': 0}
        
        # 创建环境并重置
        env = TradingEnvironment(df, feature_cols)
        state = env.reset()
        
        # 运行到最后
        while not env.done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                action = q_values.argmax().item()
            state, _, done, info = env.step(action)
        
        # 根据最终持仓给出信号
        if info.get('position', 0) > 0:
            signal = '买入'
            confidence = 0.7
        else:
            signal = '持有'
            confidence = 0.5
        
        return {
            'signal': signal,
            'confidence': confidence,
            'portfolio_value': info.get('portfolio_value', 0)
        }
    
    def save_model(self, path: str):
        """保存模型"""
        if self.policy_net is not None:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'config': self.config
            }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        
        state_size = self.policy_net.network[0].in_features if self.policy_net else 7
        self.build_model(state_size)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])

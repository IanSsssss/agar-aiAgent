from stable_baselines3 import PPO
import numpy as np

class AIAgent:
    def __init__(self, model_path="models/ppo_agar_agent.zip"):
        self.model = PPO.load(model_path)
    
    def act(self, observation):
        if isinstance(observation, dict):
            # 转换成训练时用的向量格式（你已做过）
            obs = self._dict_to_vector(observation)
        else:
            obs = observation  # 已经是 numpy array
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def _dict_to_vector(self, obs_dict):
        obs = []

        # 1. 自身信息（3维）
        player = obs_dict["player"]
        obs.extend([
            player["x"] / 5000,
            player["y"] / 5000,
            player["mass"] / 500
        ])

        # 2. 5个食物相对坐标（每个2维）= 10维
        for food in obs_dict["visible_food"][:5]:
            dx = (food["x"] - player["x"]) / 1000
            dy = (food["y"] - player["y"]) / 1000
            obs.extend([dx, dy])
        while len(obs) < 3 + 5*2:
            obs.extend([0, 0])

        # 3. 最多3个敌人信息（x, y, mass）= 9维
        for enemy in obs_dict["visible_players"][:3]:
            dx = (enemy["x"] - player["x"]) / 1000
            dy = (enemy["y"] - player["y"]) / 1000
            dm = enemy["mass"] / 500
            obs.extend([dx, dy, dm])
        while len(obs) < 3 + 10 + 9:
            obs.extend([0, 0, 0])

        return np.array(obs, dtype=np.float32)
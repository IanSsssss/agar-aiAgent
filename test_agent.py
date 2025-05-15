from stable_baselines3 import PPO
from demo import AgarEnvironment
import time

# 加载环境和模型
env = AgarEnvironment()
model = PPO.load("models/ppo_agar_agent")

obs = env.reset()
done = False
total_reward = 0
step = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    step += 1

    # 可视化调试输出
    print(f"Step: {step} | Reward: {reward:.2f} | Total Mass: {info['player_mass']:.2f}")
    env.render()
    time.sleep(0.1)  # 控制打印节奏

print(f"✅ 测试完成，总奖励: {total_reward:.2f}")

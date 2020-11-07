from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

from stable_baselines3 import DQN

from env import SnakeEnv, CurrSnakeEnv
from model import CustomCNN

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5000",
    "http://localhost:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

# Instantiate the env
s = 8
env = SnakeEnv(s)
model_file = f"models/model_S_{s}"

model = DQN.load(model_file)

total_reward = 0
obs = env.reset()


@app.get("/load")
async def load():
    global obs, total_reward
    obs = env.reset()
    total_reward = 0
    render = env.render()
    return {
        "obs": render.tolist(),
        "size": s,
    }


@app.get("/step")
async def step():
    global model, obs, total_reward
    obs_t = torch.tensor(obs)
    obs_t = torch.transpose(obs_t, 0, 2)
    logits = model.policy.q_net(obs_t.view((1, 3, 8, 8)).to(device))
    probs = F.softmax(logits, dim=1)
    probs = probs.detach().cpu().numpy()[0]
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    render = env.render()
    print([total_reward])
    print([done])
    return jsonable_encoder({
        "obs": render.tolist(),
        "total_reward": float(total_reward),
        "done": bool(done),
        "size": s,
        "probs": probs.tolist(),
    })

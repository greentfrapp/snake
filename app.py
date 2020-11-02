from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

from stable_baselines3 import DQN
from stable_baselines3.common.cmd_util import make_vec_env

from adv_env import SnakeEnv
from learn import CustomCNN

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
dim = 20
env = SnakeEnv(dim)
# wrap it
env = make_vec_env(lambda: env, n_envs=1)

model = DQN.load(f"snake_{dim}x{dim}_adv")

total_reward = 0
obs = env.reset()


@app.get("/load")
async def load():
    global model, obs, total_reward
    try:
        model = DQN.load(f"snake_{dim}x{dim}_adv")
    except:
        pass
    obs = env.reset()
    total_reward = 0
    render = env.render()
    return {"obs": render.tolist()}


@app.get("/step")
async def step():
    global model, obs, total_reward
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    render = env.render()
    print([total_reward])
    print([done])
    return jsonable_encoder({
        "obs": render.tolist(),
        "total_reward": float(total_reward[0]),
        "done": bool(done[0]),
    })

import random
from typing import List



class Environment:
    _steps_left: int


    def __init__(self):
        self._steps_left = 10


    def perform_action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over.")
        self._steps_left -= 1
        return random.random()


    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]


    def get_actions(self) -> List[int]:
        return [0, 1]


    def is_done(self) -> bool:
        return self._steps_left <= 0



class Agent:
    def __init__(self):
        self.total_reward = 0.0


    def step(self, env: Environment):
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.perform_action(random.choice(actions))
        self.total_reward += reward



if __name__ == "__main__":
    env = Environment()
    agent = Agent()
    while not env.is_done():
        agent.step(env)
    print("Total reward got: %.4f", agent.total_reward)

from typing import Any, List, Optional, Tuple

import gym
import ptan



# noinspection PyAbstractClass
class ToyEnv(gym.Env):
    def __init__(self):
        super(ToyEnv, self).__init__()

        self.observation_space = gym.spaces.Discrete(5)
        self.action_space = gym.spaces.Discrete(3)
        self.step_index = 0


    def reset(self):
        self.step_index = 0
        return self.step_index


    def step(self, action: int):
        if self.step_index == 10:
            return self.step_index % self.observation_space.n, 0.0, True, { }

        self.step_index += 1
        return self.step_index % self.observation_space.n, float(action), self.step_index == 10, { }



class DullAgent(ptan.agent.BaseAgent):
    def __init__(self, action: int):
        self._action = action


    def __call__(self, observations: List[Any], state: Optional[List[Any]]) -> Tuple[List[int], Optional[List[Any]]]:
        return [self._action for _ in observations], state



if __name__ == '__main__':
    env = ToyEnv()
    agent = DullAgent(action = 1)

    print('ExperienceSource:')
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count = 3)
    for idx, exp in enumerate(exp_source):
        # Only print three experiences (the source would generate infinitely many).
        if idx > 15:
            break
        print('', exp)

    print('ExperienceSourceFirstLast')
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma = 1.0, steps_count = 3)
    for idx, exp in enumerate(exp_source):
        # Only print three experiences (the source would generate infinitely many).
        if idx > 15:
            break
        print('', exp)

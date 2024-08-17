import gymnasium as gym
import time

class ActionNode:
    def __init__(self, action):
        self.action = action

    def step(self, env, observation):
        return env.step(self.action)
    
class PredicateNode:
    def __init__(self, predicate, lchild, rchild):
        self.predicate = predicate
        self.lchild = lchild
        self.rchild = rchild

    def step(self, env, observation):
        if self.predicate(observation):
            return self.lchild.step(env, observation)
        
        return self.rchild.step(env, observation)


def is_cart_going_right(observation):
    return observation[1] > 0

def is_cart_left_of_position(position):
    return lambda observation:observation[0] < position

def main():
    #0 - accelerate left, 1 - do no accelerate, 2 - accelerate right
    # This decision tree accelerates left until a certain point to get some momentum,
    # and then only accelerates right
    
    root = PredicateNode(is_cart_left_of_position(-0.9), 
        ActionNode(2), PredicateNode(is_cart_going_right, ActionNode(2), ActionNode(0)))

    env = gym.make('MountainCar-v0', render_mode="human")
    observation, info = env.reset(seed=42)

    total_reward = 0
    for _ in range(1000):
        observation, reward, terminated, truncated, info = root.step(env, observation)
        total_reward += reward
        if terminated or truncated:
            print("finished with reward: " + str(total_reward))
            env.close()
            return
    env.close()

if __name__ == "__main__":
    main()
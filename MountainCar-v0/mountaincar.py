import gymnasium as gym 

def run():
    env = gym.make('MountainCar-v0',render_mode='human')    
    state = env.reset()[0]
    print("intial state",state)
    terminated = False  
    rewards = 0 
    while not terminated and rewards>-100:
        action = env.action_space.sample()
        new_state, reward, terminated, _,_ = env.step(action)
        state = new_state
        rewards += reward
        print("state",state)
        print("reward",reward)
        
    env.close()
    
if __name__ == "__main__":
    run()   
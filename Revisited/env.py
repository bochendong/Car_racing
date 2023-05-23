
import numpy as np
import gym

ORG_B0 = np.array([100, 202, 100])
ORG_B1 = np.array([100, 228, 100])

Unseen_1_B0 = np.array([139, 177, 155])
Unseen_1_B1 = np.array([145, 202, 156])

Unseen_2_B0 = np.array([99, 160, 177])
Unseen_2_B1 = np.array([115, 159, 174])

def change_background_color(img_rgb, new_background_color, original_background_color,  similarity_threshold=25):
    # Calculate the absolute difference between the image and the original background color
    color_difference = np.abs(img_rgb - original_background_color)

    # Create a boolean mask to identify the background pixels
    background_mask = np.all(color_difference <= similarity_threshold, axis=-1)

    # Replace the original background color with the new background color
    img_rgb[background_mask] = new_background_color

    return img_rgb


class Env():
    def __init__(self, color, seed=0):
        """
        Initialize the environment with the given color and seed.
        
        :param color: str, color for the environment, 'g' for green
        :param seed: int, seed for the environment
        """

        self.env = gym.make('CarRacing-v2')
        self.env.reset(seed=seed)
        
        self.color = color
        self.reward_threshold = 1000
        self.reset()

    def reset(self):
        """
        Reset the environment and return the initial state.
        
        :return: np.array, initial state
        """
        self.counter = 0
        self.die = False
        self.av_r = self.reward_memory()

        img_rgb = self.env.reset()
        if (self.color == 'c1'):
            img_rgb = change_background_color(img_rgb, Unseen_1_B0, ORG_B0)
            img_rgb = change_background_color(img_rgb, Unseen_1_B1, ORG_B1)
        elif (self.color == 'c2'):
            img_rgb = change_background_color(img_rgb, Unseen_2_B0, ORG_B0)
            img_rgb = change_background_color(img_rgb, Unseen_2_B1, ORG_B1)

        img_rgb = img_rgb / 128. - 1

        self.stack = [img_rgb] * 4
        return np.array(self.stack)


    def step(self, action):
        """
        Take a step in the environment using the given action, reconstrcut the 
        reward fucntion.
        
        :param action: np.array, action to take in the environment
        :return: tuple, (state, total_reward, done, die)
        """
        total_reward = 0
        for i in range(8):
            img_rgb, reward, die, _ = self.env.step(action)

            if (self.color == 'c1'):
                img_rgb = change_background_color(img_rgb, Unseen_1_B0, ORG_B0)
                img_rgb = change_background_color(img_rgb, Unseen_1_B1, ORG_B1)
            elif (self.color == 'c2'):
                img_rgb = change_background_color(img_rgb, Unseen_2_B0, ORG_B0)
                img_rgb = change_background_color(img_rgb, Unseen_2_B1, ORG_B1)

            if die: reward += 100

            if self.color == 'g' and np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05

            total_reward += reward

            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break

        img_rgb = img_rgb / 128. - 1


        self.stack.pop(0)
        self.stack.append(img_rgb)
        assert len(self.stack) == 4
        return np.array(self.stack), total_reward, done, die

    def step_eval(self, action):
        """
        Take a step in the environment using the given action (for evaluation).
        
        :param action: np.array, action to take in the environment
        :return: tuple, (state, reward, done, _)
        """
        img_rgb, reward, done, _ = self.env.step(action)

        img_rgb = img_rgb / 128. - 1
        self.stack.pop(0)
        self.stack.append(img_rgb)
        return np.array(self.stack), reward, done, _

    def render(self, *arg):
        """
        Render the environment.
        
        :param arg: tuple, arguments for the rendering function
        """
        self.env.render(*arg)

    @staticmethod
    def reward_memory():
        """
        Create a reward memory function to store the last `length` rewards.
        
        :return: function, reward memory function
        """
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
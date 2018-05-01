import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import sklearn.pipeline
import sklearn.preprocessing
import plotting
from matplotlib import pyplot as plt
if "../" not in sys.path:
  sys.path.append("../") 

#from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import PassiveAggressiveRegressor
from lightning.regression import FistaRegressor,AdaGradRegressor, CDRegressor
from wordbatch.models import FTRL
import time

# Acribot_v1 Env
matplotlib.style.use('ggplot')
env = gym.envs.make("Acrobot-v1")
# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
#print observation_examples
# original 10000
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurized representation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=0.5, n_components=50)),
        ("rbf2", RBFSampler(gamma=1, n_components=50)),
        ("rbf3", RBFSampler(gamma=2, n_components=50)),
        ("rbf4", RBFSampler(gamma=5, n_components=50))
        ])
featurizer.fit(scaler.transform(observation_examples))
# default 100
class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self,spar_type,spar_penalty):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            #model=Lasso(alpha=0.01)
            model = SGDRegressor(learning_rate='constant',penalty=spar_type,l1_ratio=spar_penalty,max_iter=1000)
            model1 = PassiveAggressiveRegressor()
            model2 = Lasso(alpha=0.1,normalize=True,warm_start=True)
            model3= FTRL(alpha=1.0, beta=1.0, L1=0.00001, L2=1.0, D=2 ** 25, iters=1)
            #l2,l1,none,elasticnet
            #,penalty='l1',l1_ratio=0)
            #learning_rate="constant"

            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            #model2.fit([self.featurize_state(env.reset())], [0])
            #X = np.array([self.featurize_state(env.reset())])
            #Y = np.array([0])
            #model.partial_fit(X,Y)

            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        #heyhey=np.where( featurized[0] > 0.01 )

        #print 'hey+',np.count_nonzero(heyhey)
        return featurized[0]
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        #self.models[a].fit([features], [y])
        X = np.array([features])
        Y = np.array([y])
        #self.models[a].partial_fit(X,Y)
        self.models[a].partial_fit([features], [y])
        #whatheck= np.count_nonzero([features]) 
        #heyhey=np.where( features > 0.1)
        #print features

        #print 'hey+',np.count_nonzero(heyhey)

        heyheyyy=self.models[a].coef_ 
        print np.count_nonzero(np.where( heyheyyy> 0.1))
        
        #print whatheck

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1, epsilon=0.1, epsilon_decay=1.0):
    #default gamma=1
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    for i_episode in range(num_episodes):
        
        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print last_reward
        #sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # Only used for SARSA, not Q-Learning
        next_action = None
        
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            
            # Take a step
            next_state, reward, done, _ = env.step(action)
    
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            q_values_next = estimator.predict(next_state)
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
            
            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
            # td_target = reward + discount_factor * q_values_next[next_action]
            
            # Update the function approximator using our target
            estimator.update(state, action, td_target)
            #print td_target
            
            print "\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), #end=""
                
            if done:
                break
                
            state = next_state
    
    return stats
print 'start1'
estimator = Estimator('l1',1)
start_time1 = time.time()
stats = q_learning(env, estimator,500, epsilon=0.1)

end_time1 = start_time1-time.time()
#plotting.plot_cost_to_go_mountain_car(env, estimator)
print 'start2'
estimator2 = Estimator('l2',0.1)
start_time2 = time.time()
stats2 = q_learning(env, estimator2, 500, epsilon=0.1)
end_time2 = start_time2-time.time()

estimator3 = Estimator('l2',0.01)
start_time3 = time.time()
stats3 = q_learning(env, estimator3, 500, epsilon=0.1)
end_time3 = start_time3-time.time()

print end_time1,end_time2,end_time3
#plotting.plot_episode_stats(stats, smoothing_window=10)
#plotting.plot_episode_stats(stats2, smoothing_window=10)
#plotting.show()


smoothing_window=10
fig1=plt.figure(figsize=(8,5))
rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
rewards_smoothed2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
rewards_smoothed3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()


plt.plot(rewards_smoothed,'r',rewards_smoothed2,'b',rewards_smoothed3,'g')
plt.legend(('L1', 'L2', 'Elastic'))

plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
plt.title("Episode Reward over Time in Acrobot Domain")
plt.show()

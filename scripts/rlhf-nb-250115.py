import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
import pickle, random
from tqdm import tqdm
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import os

#!pip install matplotlib

indicatorfiles = [
'spreadsheets/rlhf_20250104_6.csv',
'spreadsheets/rlhf_159nlp.csv',
'spreadsheets/rlhf_1064_2.csv',
'spreadsheets/rlhf_large_1072.csv',
'spreadsheets/shufled_rlhf_11rl.csv',
'spreadsheets/rlhf_large_12rl.csv',
'spreadsheets/rlhf_large_15rl.csv',
'spreadsheets/rlhf_large_19rl.csv'
]

Hyperparameters = [
    #[0.25, 0.95, 1.0, 0.99, 0.99, 10000],
    [0.9, 0.95, 1.0, 0.99, 0.99, 8000],
    #[0.7, 0.75, 1.0, 0.97, 0.999, 28000],
    [0.05, 0.95, 1.0, 0.999, 0.995, 12000],
    #[1, 0.95, 0.1, 0.96, 0.96, 12000],
    #[0.25, 0.99, 0.01, 0.997, 0.99, 8000],
    #[1, 0.75, 0.05, 0.999, 0.999, 10000],
    [0.01, 0.85, 0.01, 0.95, 0.95, 12000],
    #[0.01, 0.99, 1.0, 0.95, 0.99, 16000],
    [0.05, 0.99, 0.5, 0.99, 0.997, 6000],
    #[0.25, 0.95, 0.01, 0.997, 0.999, 14000],
    [0.01, 0.95, 1.0, 0.997, 0.995, 26000],
    #[0.9, 0.99, 0.5, 0.995, 0.95, 12000],
    #[1, 0.75, 0.005, 0.95, 0.95, 22000],
    [0.005, 0.75, 0.1, 0.95, 0.999, 12000],
    #[0.25, 0.75, 0.01, 0.995, 0.999, 20000]
]



for item in indicatorfiles:
    if not os.path.exists(item):
        print(f'{item} not found in the file system')
    else:
        for param in Hyperparameters:

            df0 = pd.read_csv(f'{item}')

            df0['ask'] = df0['close'] * df0['volume']/(df0['close'] + df0['open'])

            df0['bid'] = df0['open'] * df0['volume']/(df0['close'] + df0['open'])

            #df0 = df0.reset_index(drop=True)

            df0.head()



            df0.drop(['Unnamed: 0'], axis=1, inplace=True)

            #df0['action'].value_counts()

            #action_mapping = {"go_long": 0, "go_short": 1}

            def prep_data(df: DataFrame) -> DataFrame:
                train_data = pd.DataFrame()
                for col in df.columns:
                    col_name = col.split(' ')[0]
                    train_data[f'{col_name}'] = df[col]

                return train_data

            train_data = df0 #prep_data(df0)

            train_data.head(2)

            new_cols = ['ask','bid','sma-compare','is_short']

            # Encode actions into numerical values
            action_mapping = {"go_long": 0, "go_short": 1, "do_nothing": 2}
            train_data["action_num"] = train_data["nlpreds"].map(action_mapping)

            # Define RL parameters
            #states = train_data[["sma-05", "sma-07", "sma-25", "sma-compare", "is_short"]].values  # Include binary_state
            states = train_data[new_cols].values
            actions = list(action_mapping.values())  # Action space
            rewards = train_data["reward"].values  # Rewards
            n_states = states.shape[0]
            n_actions = len(actions)

            # Initialize Q-table
            q_table = np.zeros((n_states, n_actions))


            alpha, gamma, epsilon, min_epsilon, decay_rate, n_episodes = param



            def create_state_index_mapping(df):
                state_to_index = {}
                for idx, row in df.iterrows():
                    state = (row['ask'], row['bid'], row['sma-compare'], row['is_short'])
                    state_to_index[state] = idx
                return state_to_index

            # Assuming 'df' is your dataframe used during training
            state_to_index = create_state_index_mapping(train_data)

            # Save the state_to_index dictionary for later use
            np.save('bids_state_to_index.npy', state_to_index)


            # Helper function to choose an action using epsilon-greedy
            def choose_action(state, epsilon):
                if np.random.uniform(0, 1) < epsilon:
                    return np.random.randint(0, n_actions)  # Explore: random action
                else:
                    return np.argmax(q_table[state])  # Exploit: best known action


            # Set random seed for reproducibility and train the loop
            np.random.seed(42)
            random.seed(42)
            # Initialize a list to store rewards per episode
            rewards_per_episode = []

            for episode in tqdm(range(n_episodes), desc="evaluating results per episode ..."):
                current_state = np.random.randint(0, n_states)  # Random initial state
                total_reward = 0  # Initialize total reward for the current episode

                while current_state < n_states - 1:
                    action = choose_action(current_state, epsilon)
                    
                    next_state = current_state + 1  # This depends on your environment logic
                    reward = rewards[next_state]

                    best_next_action = np.argmax(q_table[next_state])
                    q_table[current_state, action] += alpha * (
                        reward + gamma * q_table[next_state, best_next_action] - q_table[current_state, action]
                    )
                    
                    total_reward += reward  # Accumulate reward for the current episode
                    current_state = next_state  # Move to next state

                rewards_per_episode.append(total_reward)  # Store the total reward for the current episode

                # Decay epsilon
                epsilon = max(min_epsilon, epsilon * decay_rate)

                # Optional: Log progress
                #if episode % 400 == 0:  # Adjust logging frequency as needed
                #    print(f"Episode {episode}/{n_episodes} - Total Reward: {total_reward}, Epsilon: {epsilon}")

            # Example: Save the Q-table
            np.save("bids_q_table.npy", q_table)

            # Example: Plotting the rewards
            #import matplotlib.pyplot as plt

            plt.plot(rewards_per_episode)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Rewards per Episode')
            #plt.show()



            def load_q_table(file_path):
                return np.load(file_path)

            def load_state_index_mapping(file_path):
                return np.load(file_path, allow_pickle=True).item()

            loaded_mapping = load_state_index_mapping(file_path="bids_state_to_index.npy")
            loaded_qtable = load_q_table(file_path="bids_q_table.npy")

            #state = (row['open'], row['high'], row['ema-26'], row['ema-12'], row['low'], \
            #                 row['mean-grad-hist'], row['close'], row['volume'], row['sma-25'], \
            #                 row['long_jcrosk'], row['short_kdj'], row['sma-compare'], row['is_short'])

            def prep_state(
                            ask: float, bid: float, sma_compare: int, is_short: int
                        ):
                state = np.array([[ask, bid, sma_compare, is_short]])
                if not np.all(np.isfinite(state)):
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                return state



            def predict_action(state, q_table, state_to_index, action_mapping, default_action: str = None):
                state_tuple = tuple(state.flatten())

                state_index = state_to_index.get(state_tuple, -1)

                if not state_index == -1:
                    try:
                        q_values = q_table[state_index]
                    except ValueError as e:
                        print(e)
                        #return default_action
                else:
                    state_tuples = list(state_to_index.keys())
                    kdtree = KDTree(state_tuples)
                    distance, index = kdtree.query(state.flatten())
                    nearest_state_tuple = state_tuples[index]
                    new_state_index = state_to_index[nearest_state_tuple]
                    q_values = loaded_qtable[new_state_index]
                
                #q_values = q_table[state_index]
                best_action_index = np.argmax(q_values)
                action = [action for action, index in action_mapping.items() if index == best_action_index][0]
                return action

            # Predict an entire range
            for idx, row in train_data.iterrows():
                state = row[['ask','bid','sma-compare', 'is_short']].values
                action = predict_action(state, q_table, state_to_index, action_mapping)
                train_data.loc[idx, "predicted_action"] = action

            train_data['predicted_action'].value_counts()

            m = train_data[(train_data['predicted_action'] == 'go_long') & (train_data['reward'] > 0)]
            m['is_short'].value_counts()

            longs = dict(m['is_short'].value_counts().items()).get(0)
            print(longs)
            shorts = dict(m['is_short'].value_counts().items()).get(1)
            print(shorts)

            def logrewards():
                # Performance measures
                # Initialize cumulative rewards
                cumulative_predicted_reward = 0
                cumulative_actual_reward = 0

                # Iterate through states to calculate rewards
                for state_index in range(n_states - 1):
                    # Predicted action from Q-table
                    predicted_action = np.argmax(q_table[state_index])  # Best action for the current state
                    # Actual action from the ground truth
                    actual_action = train_data["action_num"].iloc[state_index]

                    # Get reward for predicted action only if it matches the actual action
                    if predicted_action == actual_action:
                        predicted_reward = rewards[state_index + 1]  # Reward for the correct prediction
                        cumulative_predicted_reward += predicted_reward

                    # Get actual reward for the ground truth action
                    actual_reward = rewards[state_index + 1]
                    cumulative_actual_reward += actual_reward

                    # Optional: Log progress
                    #if state_index % 100 == 0:  # Adjust logging frequency as needed
                    #    print(f"Processed state {state_index}/{n_states - 1}")
                    #    print(f"Current Predicted Reward: {cumulative_predicted_reward}")
                    #    print(f"Current Actual Reward: {cumulative_actual_reward}")

                # Print results
                #print(f"Cumulative Predicted Reward: {cumulative_predicted_reward}")
                #print(f"Cumulative Actual Reward: {cumulative_actual_reward}")

                # Optionally calculate efficiency

                efficiency = (
                    ((cumulative_predicted_reward - cumulative_actual_reward) / abs(cumulative_actual_reward)) * 100
                    if cumulative_actual_reward != 0
                    else 0
                )
                return {
                    'indicators': item.split('/')[-1],
                    'predicted reward': cumulative_predicted_reward,
                    'actual reward': cumulative_actual_reward,
                    'params': param,
                    'prediction eff': f"{efficiency:.2f}"
                }
                #print(f"Prediction Efficiency for {item.split('/')[-1]} and {param}: {efficiency:.2f}%")
                #return result
            def prGreen(skk): print(f"\033[92m {skk}\033[00m")

            prGreen(logrewards())


            # Accuracy
            correct_predictions = 0
            for state_index in range(n_states):
                predicted_action = np.argmax(q_table[state_index])  # Predicted action
                actual_action = train_data["action_num"].iloc[state_index]  # Actual action
                if predicted_action == actual_action:
                    correct_predictions += 1

            accuracy = correct_predictions / n_states
            print(f"Accuracy: {accuracy * 100:.2f}%")

            # Confusion matrix
            y_true = train_data["action_num"]  # Actual actions
            y_pred = [np.argmax(q_table[state_index]) for state_index in range(n_states)]  # Predicted actions

            cm = confusion_matrix(y_true, y_pred)
            print(f"Confusion Matrix for {item}:")
            print(cm)

            #df0['action'].value_counts()

            def action_reward(action: str, is_short: int):
                m = train_data[(train_data['predicted_action'] == f'{action}') & (train_data['is_short'] == is_short)]
                counts = m['is_short'].value_counts()
                return {
                    'counts': counts.get(is_short),
                    'total_reward': m['reward'].cumsum()[-1:].values[0],
                    'wins': len(m[m['reward'] > 0]),
                    'losses': len(m[m['reward'] <= 0])
                }

            #print(action_reward('go_long', 0)) # go_short 1

            #import matplotlib.pyplot as plt

            # Assuming you have a list of rewards for each episode
            #rewards_per_episode = [...]  # Populate this with your actual data

            plt.plot(rewards_per_episode)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Rewards per Episode')
            #plt.show()


            # Tuning

            def train_q_learning(alpha, gamma, epsilon, min_epsilon, decay_rate, n_episodes, n_states, n_actions, rewards):
                q_table = np.zeros((n_states, n_actions))
                rewards_per_episode = []

                for episode in range(n_episodes):
                    current_state = np.random.randint(0, n_states)
                    total_reward = 0

                    while current_state < n_states - 1:
                        if np.random.uniform(0, 1) < epsilon:
                            action = np.random.randint(0, n_actions)
                        else:
                            action = np.argmax(q_table[current_state])

                        next_state = current_state + 1  # Adjust based on environment logic
                        reward = rewards[next_state]

                        best_next_action = np.argmax(q_table[next_state])
                        q_table[current_state, action] += alpha * (
                            reward + gamma * q_table[next_state, best_next_action] - q_table[current_state, action]
                        )

                        total_reward += reward
                        current_state = next_state

                    rewards_per_episode.append(total_reward)
                    epsilon = max(min_epsilon, epsilon * decay_rate) # default min_epsilon = 0.01

                return q_table, rewards_per_episode



            def evaluate_q_learning_reward_weighted(q_table, n_states, train_data, rewards):
                correct_predictions = 0
                total_reward = 0
                reward_weighted_accuracy = []

                # Add tqdm for progress visualization
                for state_index in tqdm(range(n_states), desc="Evaluating States"):
                    predicted_action = np.argmax(q_table[state_index])  # Predicted action
                    actual_action = train_data["action_num"].iloc[state_index]  # Actual action
                    reward = rewards[state_index]  # Reward for the action

                    if predicted_action == actual_action:
                        correct_predictions += 1
                        total_reward += reward

                    accuracy = correct_predictions / (state_index + 1)
                    reward_weighted_accuracy.append(total_reward / (state_index + 1))

                    # Optional: Log progress
                    if state_index % 100 == 0:
                        print(f"Processed state {state_index}/{n_states} - Accuracy: {accuracy * 100:.2f}%, Reward-weighted Accuracy: {reward_weighted_accuracy[-1]}")

                final_reward_weighted_accuracy = total_reward / n_states
                return final_reward_weighted_accuracy * 100



            '''
            def random_search_reward_weighted(n_iter, param_grid, n_states, n_actions, rewards, train_data):
                best_params = None
                best_reward_weighted_accuracy = float('-inf')

                for _ in tqdm(range(n_iter), desc="Searching for params ..."):
                    alpha = random.choice(param_grid['alpha'])
                    gamma = random.choice(param_grid['gamma'])
                    epsilon = random.choice(param_grid['epsilon'])
                    min_epsilon = random.choice(param_grid['decay_rate'])
                    decay_rate = random.choice(param_grid['decay_rate'])
                    n_episodes = random.choice(param_grid['n_episodes'])

                    q_table, _ = train_q_learning(alpha, gamma, epsilon, min_epsilon, decay_rate, n_episodes, n_states, n_actions, rewards)
                    reward_weighted_accuracy = evaluate_q_learning_reward_weighted(q_table, n_states, train_data, rewards)

                    if reward_weighted_accuracy > best_reward_weighted_accuracy:
                        best_reward_weighted_accuracy = reward_weighted_accuracy
                        best_params = (alpha, gamma, epsilon, min_epsilon, decay_rate, n_episodes)

                    print(f"Iteration Reward-weighted Accuracy: {reward_weighted_accuracy:.2f}%, Best Reward-weighted Accuracy: {best_reward_weighted_accuracy:.2f}%")

                return best_params, best_reward_weighted_accuracy

            # Define the parameter grid
            param_grid = {
                'alpha': [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9, 1],
                'gamma': [0.75, 0.85, 0.9, 0.95, 0.99],
                'epsilon': [1.0, 0.5, 0.1, 0.05, 0.01, 0.005],
                'min_epsilon': [0.05, 0.01, 0.005, 0.001, 0.001],
                'decay_rate': [0.95, 0.99, 0.995, 0.997, 0.999],
                'n_episodes': [1500, 4000,6000, 8000, 9000, 10000, 11000, 12000]
            }

            # Perform Random Search
            best_params, best_reward_weighted_accuracy = random_search_reward_weighted(50, param_grid, n_states, n_actions, rewards, train_data)
            print(f"Best Hyperparameters: {best_params}")
            print(f"Best Reward-weighted Accuracy: {best_reward_weighted_accuracy:.2f}%")

            '''

            def evaluate_q_learning_prediction_efficiency(q_table, n_states, train_data, rewards):
                # Initialize cumulative rewards
                cumulative_predicted_reward = 0
                cumulative_actual_reward = 0

                # Iterate through states to calculate rewards
                for state_index in range(n_states - 1):
                    # Predicted action from Q-table
                    predicted_action = np.argmax(q_table[state_index])  # Best action for the current state
                    # Actual action from the ground truth
                    actual_action = train_data["action_num"].iloc[state_index]

                    # Get reward for predicted action only if it matches the actual action
                    if predicted_action == actual_action:
                        predicted_reward = rewards[state_index + 1]  # Reward for the correct prediction
                        cumulative_predicted_reward += predicted_reward

                    # Get actual reward for the ground truth action
                    actual_reward = rewards[state_index + 1]
                    cumulative_actual_reward += actual_reward
                return cumulative_predicted_reward



            '''
            def random_search_prediction_efficiency(n_iter, param_grid, n_states, n_actions, rewards, train_data):
                best_params = None
                best_cumulative_pred_reward = float('-inf')

                for _ in tqdm(range(n_iter), desc="Searching for params ..."):
                    alpha = random.choice(param_grid['alpha'])
                    gamma = random.choice(param_grid['gamma'])
                    epsilon = random.choice(param_grid['epsilon'])
                    min_epsilon = random.choice(param_grid['decay_rate'])
                    decay_rate = random.choice(param_grid['decay_rate'])
                    n_episodes = random.choice(param_grid['n_episodes'])

                    q_table, _ = train_q_learning(alpha, gamma, epsilon, min_epsilon, decay_rate, n_episodes, n_states, n_actions, rewards)
                    cumulative_pred_reward = evaluate_q_learning_prediction_efficiency(q_table, n_states, train_data, rewards)

                    if cumulative_pred_reward > best_cumulative_pred_reward:
                        best_cumulative_pred_reward = cumulative_pred_reward
                        best_params = (alpha, gamma, epsilon, min_epsilon, decay_rate, n_episodes)

                    print(f"Iteration cumulative predicted reward: {cumulative_pred_reward:.2f}%, Best cumulative predicted reward: {best_cumulative_pred_reward:.2f}%")

                return best_params, best_cumulative_pred_reward

            # Define the parameter grid
            param_grid = {
                'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9, 1],
                'gamma': [0.75, 0.85, 0.9, 0.95, 0.99],
                'epsilon': [1.0, 0.5, 0.1, 0.05, 0.01, 0.005],
                'min_epsilon': [0.05, 0.01, 0.005, 0.001, 0.001],
                'decay_rate': [0.95, 0.99, 0.995, 0.997, 0.999],
                'n_episodes': [1500, 4000,6000, 8000, 9000, 10000, 11000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000]
            }
            # 1500, 4000,6000, 8000, 9000, 10000, 11000, 
            # Perform Random Search
            best_params, best_cumulative_pred_reward = random_search_prediction_efficiency(50, param_grid, n_states, n_actions, rewards, train_data)
            print(f"Best Hyperparameters: {best_params}")
            print(f"Best cumulative predicted reward: {best_cumulative_pred_reward:.2f}%")
            '''

#print(logrewards())


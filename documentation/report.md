## **Introduction**

This project demonstrates the application of Deep Reinforcement Learning (DRL) in stock trading, using Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), and a Feedforward Neural Network (FNN) for price prediction. A Genetic Algorithm (GA) is used to optimize the portfolio based on objectives like maximizing returns or minimizing drawdown. The project aims to create a data-driven approach that outperforms traditional trading strategies and alternative machine learning techniques.

The data consists of historical stock prices and volumes from various sectors, augmented with derived metrics such as moving averages and volatility indicators. The GA selects a subset of stocks found to be the most optimal for a specified objective, and the DRL agent learns optimal trading strategies specific to this optimized portfolio.

The goal is to develop a robust system that combines the GA for portfolio optimization with the FNN and DRL algorithms for trading decisions. The hypothesis is that this approach can adapt to market dynamics, make intelligent decisions, and produce an optimal portfolio to interact with. The following sections will detail the technical implementation, discuss challenges and strategies, and evaluate the system's performance through rigorous testing and analysis, highlighting the potential of combining evolutionary optimization and deep reinforcement learning in revolutionizing stock trading.

<br>

## **Dataset Summary**

Our dataset consisted of historical stock data [source] and derived technical indicators for a diverse range of stocks over nearly two-decades. The dataset included 35 variables, which were a combination of original stock price data and augmented variables engineered to enhance the predictive capabilities of our Feedforward Neural Network and Deep Reinforcement Learning (DRL) models.

The original variables included:
•	timestamp (int64): The timestamp of each stock price data point.
•	open, high, low, close (float64): The opening, highest, lowest, and closing prices of the stock for each timestamp.
•	volume (float64): The trading volume of the stock for each timestamp.
•	ticker (object): The ticker symbol of the stock.

The augmented variables were derived from the original stock price data and included:
•	returns (float64): The percentage change in the opening stock price from the previous timestamp.
•	avg_price, avg_returns (float64): The average price and returns over a specified window.
•	volatility, volume_volatility (float64): Measures of price and volume fluctuations.
•	moving_avg, price_change, volume_change (float64): Moving averages and changes in price and volume.
•	rolling_mean, rolling_std (float64): Rolling mean and standard deviation of the stock price.
•	rsi, ema, macd, sma_5, sma_10, sma_20, sma_50 (float64): Various technical indicators.
•	bbands_upper, bbands_middle, bbands_lower (float64): Bollinger Bands indicators.
•	vwap, roc, atr, cci, williamsr (float64): Various technical indicators.
•	stochastic_slowk, stochastic_slowd, mfi (float64): Stochastic and Money Flow Index indicators.
•	day_of_week (int64): The day of the week corresponding to each timestamp.
•	is_holiday (int64): A binary variable representing a boolean and indicating whether the timestamp falls on a holiday.

During the exploratory data analysis, we encountered some issues, such as missing data. In most instances, missing data was as a result of the stock being delisted and no longer traded. Due to DRL models learning from the entirety of the state space, the most practical way of handling missing data was to remove. This was also the case for stocks being listed later than the beginning timestamp. While this data holds training value, future work should be focused on research aiming to solve for extracting this value. The most reasonable method to resolve is to backfill and forwardfill with a monetary price of zero.

The original variables, such as price and volume data, were directly related to our project goal of developing a DRL model for stock trading. These variables provided the foundation for the model to learn patterns and make trading decisions. The augmented variables, such as technical indicators, offered additional insights into market trends, momentum, and potential reversal points, which enhanced the model's predictive learning. We found significant correlations among the variables, particularly between price-related variables (e.g., open, high, low, close) and volume. Strong correlations were also observed between the original and augmented variables, as the latter were derived from the former.

<br>

## **Background Information**

Stock trading has been a domain of significant interest for both academic researchers and financial institutions. The goal of maximizing returns while minimizing risk has driven the development of various methods and technologies to predict market movements and make informed trading decisions. Our project focuses on the application of Deep Reinforcement Learning (DRL) in stock trading, aiming to create an autonomous system that can learn from historical data and adapt to changing market conditions.

Traditionally, stock trading strategies have relied on fundamental analysis, technical analysis, and human expertise. Fundamental analysis involves evaluating a company's financial health, market position, and growth prospects to determine its intrinsic value. Technical analysis, on the other hand, focuses on studying historical price and volume data to identify patterns and trends that may indicate future price movements. Human traders use a combination of these approaches, along with their experience and intuition, to make trading decisions.

Only in recent years has machine learning techniques been increasingly applied to stock trading. Some of the popular methods include:
    1.	Supervised Learning: Techniques such as Support Vector Machines (SVM), Random Forests, and Artificial Neural Networks (ANN) have been used to predict stock prices based on historical data and various features.
    2.	Unsupervised Learning: Clustering algorithms like K-means and Self-Organizing Maps (SOM) have been employed to identify patterns and group stocks with similar characteristics.
    3.	Time Series Analysis: Models such as Autoregressive Integrated Moving Average (ARIMA) and Long Short-Term Memory (LSTM) networks have been utilized to capture temporal dependencies in stock price data and make predictions.

However, these methods have limitations in capturing the complex dynamics of the stock market and adapting to changing market conditions. Specifically, these techniques fail to consider the entirety of the state space, where all other stocks and their corresponding patterns should be considered. This is where Deep Reinforcement Learning provides significant alpha. DRL combines deep learning with reinforcement learning, enabling an agent to learn optimal actions through trial and error interactions with a pre-defined, structured environment. In the context of stock trading, the agent (our DRL model) observes the state of the market (e.g., stock prices, technical indicators) and takes actions (e.g., buy, sell, hold) to maximize a reward signal (e.g., portfolio value -> profit). The agent learns from its experiences and adjusts its strategy over time to improve its performance.

The key advantage of DRL is its ability to learn and adapt in a dynamic and uncertain environment like the stock market. By continuously updating its policy based on the rewards it receives, the DRL agent can discover profitable trading strategies that may not be apparent to human traders or traditional machine learning approaches.

Several academic research papers and commercial projects have explored the use of DRL in stock trading. For example:

1.	"Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy" by Yang et al. (2020) proposed an ensemble strategy that combines multiple DRL algorithms to improve the robustness and profitability of stock trading.
2.	"Algorithmic Trading with Deep Reinforcement Learning" by Zhang et al. (2019) developed a DRL-based system that learns to trade in a simulated environment and achieved significant returns compared to benchmark strategies.
3.	"Enhancing Trading Strategies with Deep Reinforcement Learning" by Meng et al. (2019) demonstrated the effectiveness of DRL in capturing complex market dynamics and generating profitable trading signals.
4.	Commercial platforms like Kavout and Alpaca are leveraging DRL to provide AI-powered trading solutions to investors and traders.

The success of these projects and the growing interest in DRL for stock trading highlight the potential of this approach to revolutionize the field. By harnessing the power of deep learning and reinforcement learning, our project aims to develop a robust and adaptive trading system that can navigate the complexities of the stock market and deliver superior returns.

<br>

## **Experimental Methods**

In our project, we employ a combination of Deep Reinforcement Learning (DRL), a Feedforward Neural Network (FNN), and a Genetic Algorithm (GA) for stock trading, price prediction, and portfolio optimization, respectively. The DRL model is responsible for making trading decisions based on market conditions, while the FNN model is used to predict future stock prices, which serve as additional input to the DRL model. Finally, the GA is responsible for structuring the portfolio for optimal performance based on the trading objective.

<br>

**Genetic Algorithm (GA) for Portfolio Optimization:**

The GA was used to select a subset of stocks from a larger pool of stocks based on a predefined objective. The objective could be one of the following:

•	Maximizing returns: The GA aimed to find a portfolio that maximized the total returns over a given time period.
•	Minimizing drawdown: The GA aimed to find a portfolio that minimized the maximum drawdown, which is the maximum percentage decline from a historical peak.
•	Maximizing returns while minimizing drawdown: The GA aimed to find a portfolio that achieved a balance between maximizing returns and minimizing drawdown.

The GA followed a standard evolutionary process:

1.	Initialization: A population of individual portfolios was randomly initialized, where each individual represented a subset of stocks.
2.	Fitness Evaluation: Each individual (subset of stocks) in the population (all stocks considered) was evaluated based on the selected objective using historical stock data. The fitness function calculated the returns and drawdown of each portfolio.
3.	Selection: A selection mechanism, such as tournament selection, was used to select the fittest individuals from the population to serve as parents for the next generation.
4.	Crossover: The selected parents underwent a crossover operation to create offspring portfolios. A single-point crossover was used, where a random point was chosen, and the stocks from the two parents were combined to form two new offspring.
5.	Mutation: Each offspring underwent a mutation operation with a certain probability. Mutation involved replacing a randomly selected stock in the portfolio with another stock from the larger pool.
6.	Termination: The GA iteratively performed steps 2-5 for a specified number of generations or until a satisfactory solution was found.

The GA's implementation was encapsulated in the GeneticAlgorithm class, which took the following parameters:

•	objective: The optimization objective (e.g., 'max_returns', 'min_drawdown', 'max_returns_min_drawdown').
•	population_size: The size of the population in each generation.
•	num_generations: The number of generations to evolve.
•	mutation_rate: The probability of mutation for each offspring.
•	time_interval: The time interval of the stock data (e.g., '1min', '5min', '1day').
•	start_date and end_date: The start and end dates of the historical data.
•	ticker_list: The list of ticker symbols representing the larger pool of stocks.

The time_interval, start_date, and end_date variables were all aligned with the inputs used for the FNN and DRL training. The GA returned the best individual portfolio found, along with its corresponding returns and drawdown. This constrained portfolio was then used for downstream training and trading.

<br>

**Feedforward Neural Network (FNN):**

The FNN model architecture consists of an input layer, multiple hidden layers, and an output layer. The notable architectural design choices include:

•	Input layer size: Determined by the number of features in the input data.
•	Hidden layer sizes: Defined in the hidden_sizes parameter, allowing for flexibile hyperparameter optimization in the number and size of hidden layers.
•	Output layer size: Set to 1, as the model predicts a single value (future stock price).
•	Dropout regularization: Applied after each hidden layer to prevent overfitting, with a default dropout rate of 0.5.
•	Batch normalization: An optional parameter that can be set and applied after each hidden layer to improve training stability and convergence.
•	Activation function: ReLU (Rectified Linear Unit) is used as the default activation function for the hidden layers.

The FNN model training procedure involves the following steps:

1.	Data splitting: The historical stock data is auto-downloaded and split into training and validation sets if a dataframe is not provided.
2.	Model initialization: The FNN model is instantiated with the architecture.
3.	Loss function: Huber Loss, offering a balance between Mean Absolute Error (MAE) and Mean Squared Error (MSE), is used as the loss function to measure the difference between predicted and actual stock prices.
4.	Optimizer: The Adam optimizer is used to update the model's weights during training.
5.	Training loop: The model is trained for a number of epochs, with a defined batch size. The training data is passed through the model, and the loss is computed. The optimizer adjusts the model's weights to minimize the loss.
6.	Validation: After each epoch, the performance of the model is evaluated on the validation set to monitor its ability to generalize on novel, unseen data.

Hyperparameter tuning and architectural adjustments are performed using Optuna to optimize the FNN model's performance. Optuna efficiently searches the hyperparameter space and finds the best combination of learning rate and hidden layer sizes that minimize the validation loss. The optimal hyperparameters are then used to train the final FNN model.

<br>

**Deep Reinforcement Learning (DRL):**

We experimented with two popular Deep Reinforcement Learning (DRL) algorithms for stock trading: Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO). Both algorithms are designed to learn optimal trading strategies by interacting with the market environment and adapting to changing market conditions. After obtaining the optimized portfolio from the GA, the selected subset of stocks was used to train the DRL agents (SAC and PPO) to learn optimal trading strategies. The DRL agents were trained using the same experimental setup and architectural choices as previously described.

The key difference here is that the DRL agents were trained specifically on the optimized portfolio returned by the GA. This focused the agents' learning on a more promising subset of stocks, potentially leading to better trading performance and less computationally expensive training.

<br>

*Soft Actor-Critic (SAC):* SAC is an off-policy DRL algorithm that combines the benefits of both value-based and policy-based methods. The SAC architecture consists of two main components: the actor network and the critic network.

The actor network, implemented in the ActorSAC class, is responsible for selecting actions (trading decisions) based on the current state of the market. It takes the state as input and outputs the mean and log-standard deviation of a Gaussian distribution. The action is then sampled from this distribution using reparameterization, allowing for the learning of a stochastic policy. The critic network, implemented in the CriticSAC class, estimates the Q-values of state-action pairs. It takes the state and action as input and outputs two Q-value estimates using separate neural networks. The use of two Q-value estimates helps to stabilize the learning process and mitigate overestimation bias.

Notable architectural design choices in the SAC implementation include:

•	The actor and critic networks are built using fully connected layers (MLPs) with ReLU activation functions.
•	The output of the actor network is split into mean and log-standard deviation, which are used to parameterize a Gaussian distribution for action sampling.
•	The critic network outputs two Q-value estimates to improve stability and reduce overestimation bias.

The SAC algorithm follows a specific training procedure:

1.	The agent interacts with the environment and collects experiences (states, actions, rewards, next states) for a specified horizon length.
2.	The collected experiences are used to update the critic network by minimizing the mean squared error between the predicted Q-values and the target Q-values.
3.	The actor network is updated using the critic network's Q-values to maximize the expected future rewards.
4.	The target networks for the critic are updated using a soft update mechanism to stabilize learning.

Hyperparameters such as the learning rate, discount factor (gamma), and the entropy coefficient (alpha) were tuned to optimize the performance of the SAC algorithm.

<br>

*Proximal Policy Optimization (PPO):* PPO is an on-policy DRL algorithm that has shown impressive performance in various domains, including stock trading. The PPO architecture also consists of an actor network and a critic network. 

The actor network in PPO is responsible for selecting actions based on the current state. It takes the state as input and outputs the mean and log-standard deviation of a Gaussian distribution, similar to SAC. The action is then sampled from this distribution. The critic network in PPO estimates the value of each state. It takes the state as input and outputs a single value estimate.

The PPO algorithm follows a specific training procedure:

1.	The agent interacts with the environment and collects experiences (states, actions, rewards, next states) for a specified horizon length.
2.	The collected experiences are used to update the actor and critic networks.
3.	The actor network is updated using the PPO objective, which aims to maximize the expected future rewards while constraining the policy update to prevent large deviations from the previous policy.
4.	The critic network is updated by minimizing the mean squared error between the predicted state values and the target state values.

Hyperparameters such as the learning rate, discount factor (gamma), and the clip range for the PPO objective can be adjusted to optimize the performance of the PPO algorithm.

<br>

In our experiments, we trained both the SAC and PPO agents using a rollout buffer to store the collected experiences. The agents interact with the stock market environment for a specified number of iterations, and the networks are updated using the collected experiences.

We optimized the models by tuning various hyperparameters, such as the learning rate, discount factor, and network architectures (e.g., number of hidden layers and units). We also experimented with different reward scaling techniques and exploration strategies to improve the agents' performance. The trained DRL agents were then evaluated on a separate test dataset to assess their ability to generate profitable trading strategies in unseen market conditions. The performance metrics, such as cumulative returns and Sharpe ratio, were used to compare the effectiveness of the SAC and PPO algorithms.


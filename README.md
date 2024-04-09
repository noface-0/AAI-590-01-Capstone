# AAI-590-01-Capstone
# Team 6
## Nathan Metheny, Javon Kitson, Adam Graves
# AI Stock Trading System
    @misc{
      author = {Nathan Metheny, Javon Kitson, Adam Graves},
      title = {AI Smart Stock Trading System},
      year = {2024},
      project = {AAI590-Capstone},
      journal = {GitHub repository},
      howpublished = {\url(https://github.com/noface-0/AAI-590-01-Capstone/edit/main)},
    }
-------------------------------------------------------------------------------------------------------------
## Company and System Overview
### We offer a SaaS based subscription services that provides end users the ability to set up their trading portfolio and automate their stock trading utilizing our AI based system
---------------------------------------------------------------------------------------
## Project Background
The objective is to build a trading system that at its core are our AI models based on Deep Reinforcement Learning (DRL) in the domain of stock trading. By leveraging advanced machine learning techniques, we aim to develop a system capable of making informed trading decisions autonomously, based on the investor's portfolio. This involves the creation of a model that is traind by many metrics that produce a system that can adhere to the investor's portfolio, be compliant, and execute trades with the goal of maximizing returns. The demonstration will cover the setup, training, monitoring, and evaluation of the DRL model, showcasing its potential to outperform traditional trading strategies.

In addition to the technical implementation, we will explore the theoretical foundations of reinforcement learning and its suitability for financial markets. This enhances the traditional models that preditct the estimated stock trading price, and with additional metrics it will produce better predictions as to trade based on size, market time, market status, and more. This also includes discussing the challenges of applying DRL in a highly volatile environment, such as stock trading, and the strategies used to mitigate these risks. Our demonstration aims to provide a comprehensive overview of how deep reinforcement learning can be utilized to innovate in the field of stock trading, offering insights into both its capabilities and limitations.

## Data Sources
- Stock data
The dataset used in this Deep Reinforcement Learning Stock Trading is a paid a licensed dataset from Firstrate Data. 
Data Information:
FREQUENCY	DATE RANGE	NUMBER OF TICKERS	
1-minute,
5-minute,
30-minute,
1-hour
1-day	Jan 2005 - Mar 2024
(see below dates for each ticker)	10120 Tickers	

- Portfolio Data
Data input per client:
### Portfolio Fields	dtypes	Notes
name	object	
social_security_number_or_taxpayer_identification_number	int	
address	object	
telephone_number	int	
email	object	
dob	object	Calculate age. 21-29=0, 30-39=1, 40-49=2, 50-59=3, 60-68=4, 69+=5  (0-5)
id	object	
employment_status	boolean	0=unemployed 1=employed
whether_you_are_employed_by_a_brokerage_firm	boolean	0=no 1=yes
annual_income 	int	if <100000 then 'annual_income_score'=0 if >=100000 and <300000 then = 1 and if >=300000 but < 600000 then = 2 and if >=600000 then = 3
other_investments	int	
financial_situation	object	
tax_status	object	Selection M, W, S, D (S=0, M=1, D=2, W=3)
investment_experience_and_objectives	object	N=0, S=1, M=2
investment_time_horizon	object	short=0, medium=1, long=2, none=2
liquidity_needs_and_tolerance_for_risk	int	Thisis what we are working for the model. It will get updated by the calculations of other fields (in yellow) 0 to 30
financial_and_trading_record	object	
net_worth	int	0-10
trading_experience	int	N=0, S=1, M=2 (N=none, S=some, M=much)
financial_knowledge 	int	N=none(0), M=Medium(1), G=Good(2)
		
Logic networth		 (0-10): 100,000 = 0, 200,000=1 etc. 1mm+=10
annual_income_score		based off the annual_income
![image](https://github.com/noface-0/AAI-590-01-Capstone/assets/139398917/9b7e25a4-a4f6-4fd1-b7a6-35ae787ec8d2)


### Data Correlation
![Heatmap-2](https://github.com/noface-0/AAI-590-01-Capstone/assets/139398917/d8c184ae-c266-43dc-9055-41f7de44e237)

## Models
- Soft Actor-Critic (SAC1):
    - Is responsible for selecting actions (trading decisions) based on the current state of the market. 
- Proximal Policy Optimization (PPO):
    - is responsible for selecting actions based on the current state. It takes the state as input and outputs the mean and log-standard deviation of a Gaussian distribution, similar to SAC.    
 - Feedforward Neural Network (FNN):
    - is used to predict future stock prices, which serves as additional input to the DRL model. 
- Genetic Algorithm (GA):
    - is responsible for structuring the portfolio for an individual with optimal performance based on the trading objective.

## Project Management Flow and Tool
- Using Jira - Kanban Method
![image](https://github.com/noface-0/AAI-590-01-Capstone/assets/139398917/c1e30227-6ad1-4c20-9f36-631acbd7605d)

## Operations
____________________________________________________________


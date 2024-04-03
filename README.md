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
    - is responsible for selecting actions based on the current state. It takes the state as input and outputs the mean and log-standard deviation of a Gaussian     
    distribution, similar to SAC. 
- Feedforward Neural Network (FNN):
    - is used to predict future stock prices, which serves as additional input to the DRL model. 
- Genetic Algorithm (GA):
    - is responsible for structuring the portfolio for an individual with optimal performance based on the trading objective.

## Project Management Flow and Tool
- Using Jira - Kanban Method
![image](https://github.com/noface-0/AAI-590-01-Capstone/assets/139398917/c1e30227-6ad1-4c20-9f36-631acbd7605d)

## Operations
____________________________________________________________
## Team meeting #1 Date: 3/3/2024
- Team intoductions
- Dataset selection
- Brainstorm project creation and business simulation
- First task assignments
  
## Team meeting #2 Date 3/7/2024
Index (D=Discussion, B=Brainstorm, A=Adam, J=Javon, N=Nathan)
- Overview of main model for trading (D)
- Short discussion of the model algo: SAC (D)
- Metrics used for the model (D)
- S3-Athena-Feature Store relations (D)
- Monitor to be used in the model : RAM & CPU utilization (D)
- CI/CD for end-of-day kick off of model retraining (D)
- Model enhancements to include the limit price, and stock sector as metrics (B)
- Portfolio flow logic: identify fields (A)
- Enhanced Readme file details (J)
- Main model code into new repository (N)
- Review Portfolio flow (J, N)
- Start project documentation (A)
- Get Project Management system going (A,J,N)
- Setup next meeting : 3/14/24

## Team meeting #3 Date 3/14/2024
- Updates on progress
- Discussion about the Project Management tool Jira: Best Practices (A)
- Work done on the dataset as related to the scrubbing, transformation,and feature engineering (J)
- Work done on the MRL model (N)
- What is needed on the Portfolio task (N, A)
- Adam to work on the code for the Portfolio input
- Nathan and Javon to go over the logistics and fields for the Portfolio
- decided to do morning written standup emulation with short written updates and if any obstacles
- Setup meeting with Professor Marbut @ Tue 3/19/2024 3:00 PM (PST)
- Setup next Team meeting 3/21/24 @ 3:00 PST

## Team meeting #4 Date 3/21/2024
- Update progress
- Documentation discussion and presentations
- ETL services
- Feature Engineering
- SAC, PPO, FNN, and GA models
- Identify the 3 risk factors that the GA will work by
- FNN discussion on portfolio integration
- Portfolio risk related logistics
- project management updates
- Setup next Team meeting 3/28/24 @ 3:00 PST
  
## Team meeting #5 Date 3/28/2024
- Updated progress
- Go over the code review for the SAC, PPO, and FNN
- Go over the code for the portfolio input screen - create the input_data file
- Discuss the calculations of the risk factor
- Related to the risk factor discuss how the GA will tie into the input_data to pickup the risk factor
- Setup next Team meeting 4/4/24 @ 3:00 PST

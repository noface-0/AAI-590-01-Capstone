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
Data imput per client

### Data Correlation
![Heatmap-2](https://github.com/noface-0/AAI-590-01-Capstone/assets/139398917/d8c184ae-c266-43dc-9055-41f7de44e237)

## Models
Soft Actor-Critic (SAC1)
Convolutional Neural Network (CNN)
Deep Neural Network (DNN)

## Project Management Flow and Tool
- Using Jira
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

## Team meeting #4 3/21/2024
- Update progress
- Documentation discussion and presentations
- ETL services
- Feature Engineering
- SAC, CNN, and DNN models
- FNN discussion on portfolio integration
- Portfolio risk related logistics
- project management updates
  


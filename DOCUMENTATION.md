# Deep Reinforcement Learning Stock Trading Demonstration
<br>

### Introduction

Our objective is to demonstrate the application of Deep Reinforcement Learning (DRL) in the domain of stock trading. By leveraging advanced machine learning techniques, we aim to develop a system capable of making informed trading decisions autonomously. This involves the creation of a model that can analyze historical stock data, understand market trends, and execute trades with the goal of maximizing returns. The demonstration will cover the setup, training, and evaluation of the DRL model, showcasing its potential to outperform traditional trading strategies.

In addition to the technical implementation, we will explore the theoretical foundations of reinforcement learning and its suitability for financial markets. This includes discussing the challenges of applying DRL in a highly volatile environment, such as stock trading, and the strategies used to mitigate these risks. Our demonstration aims to provide a comprehensive overview of how deep reinforcement learning can be utilized to innovate in the field of stock trading, offering insights into both its capabilities and limitations.


### Dataset

The dataset used in this Deep Reinforcement Learning Stock Trading Demonstration comprises historical stock data, including open, high, low, close prices, and volume for a selection of stocks over a specified period. This data is essential for training our DRL model, allowing it to learn and make predictions about future stock movements based on past trends. The dataset includes a diverse range of stocks from various sectors, ensuring a comprehensive learning experience for the model. The period covered by the dataset is from [start_date] to [end_date], encompassing [number_of_stocks] stocks. We have structured the system to train and trade on a subset of the dataset that corresponds with the predefined groupings within the stock market (SP 100, DOW 30, NAS 100, etc.). This rich dataset serves as the foundation for our demonstration, enabling the DRL model to simulate trading strategies and evaluate their performance in a controlled, paper trading environment. To enhance the predictive capabilities of our Deep Reinforcement Learning (DRL) model and provide a more nuanced understanding of the stock market dynamics, we augment select variables within our dataset. This augmentation process involves creating new variables or modifying existing ones to include derived metrics, such as moving averages, volatility indicators, and technical analysis signals. These augmented variables offer additional insights into market trends, momentum, and potential reversal points, which are crucial for making informed trading decisions.

By incorporating these augmented variables, our DRL model can analyze not only the raw historical stock data but also the underlying patterns and relationships between different market indicators. This enriched dataset allows the model to capture a more comprehensive picture of the market, leading to improved accuracy in predicting future stock movements and optimizing trading strategies. The augmentation process is carefully designed to ensure that the new variables are relevant, non-redundant, and contribute positively to the model's learning process, thereby enhancing its overall performance in the stock trading demonstration.

<p align="center">
  <img src="assets/dataset_head.png" alt="Deep Reinforcement Learning Dataset">
</p>

<p align="center">
  <img src="assets/dataset_augmented.jpeg" alt="Deep Reinforcement Learning Dataset Augmented">
</p>



### Setup

The setup phase is crucial for the successful implementation of our Deep Reinforcement Learning agent. It involves the establishment of a feature store and the creation of feature groups, which are essential for organizing and managing the data our model will learn from. A feature store is a centralized repository for storing, retrieving, and managing machine learning features. Within this store, feature groups are defined to categorize and isolate different sets of features relevant to specific aspects of stock trading, such as price trends, volume changes, and market sentiment. These groups facilitate efficient data handling and model training by structuring the dataset in a way that is both accessible and meaningful for the DRL model. By meticulously setting up our feature store and carefully defining our feature groups, we lay the groundwork for a robust and scalable machine learning pipeline that is primed for the complex task of stock trading. This setup not only streamlines the model training process but also enhances the model's ability to learn from a rich and diverse dataset, ultimately contributing to more informed and effective trading decisions. In the following sections, we will delve into the specifics of how these components are integrated into our system, accompanied by illustrative diagrams and screenshots to provide a clear understanding of the setup process.

<p align="center">
  <img src="assets/athena_tables.png" alt="Athena Tables" style="width: 90%;">
</p>
<p align="center">
<img src="assets/feature_store.png" alt="Feature Store" style="width: 49%;">
  <img src="assets/features.png" alt="Feature Store" style="width: 49%;">
</p>



### Infrastructure Monitoring Dashboards

To ensure the smooth operation and reliability of our Deep Reinforcement Learning Stock Trading system, implementing comprehensive infrastructure monitoring dashboards is paramount. These dashboards serve as the nerve center for our system's health, providing real-time insights into various metrics and performance indicators critical to both the infrastructure and the application layers. By leveraging these dashboards, our team can proactively identify and address potential issues before they escalate, ranging from server load imbalances and memory leaks to latency spikes in data processing.

Our infrastructure monitoring setup encompasses a wide array of components, including compute resources, databases, network throughput, and application services. Each component is meticulously tracked to gauge its performance, availability, and resource utilization. For instance, compute resources are monitored for CPU and GPU usage, memory consumption, and disk I/O operations, ensuring that the trading model operates within optimal parameters. Similarly, network metrics provide insights into data transfer rates and latency, critical for timely and accurate trade execution.

The dashboards are designed with clarity and accessibility in mind, featuring intuitive interfaces that allow for easy navigation and interpretation of data. Alerts and notifications are configured to automatically inform the team of any anomalies or performance degradation, facilitating swift response and resolution. This proactive monitoring strategy not only minimizes downtime but also optimizes the system's performance, ensuring that the trading operations run smoothly and efficiently.

In the subsequent sections, we will explore the specific tools and technologies employed to build these dashboards, along with step-by-step guides on setting up alerts and interpreting the data presented. Through detailed examples and screenshots, readers will gain a comprehensive understanding of how to effectively monitor and maintain the infrastructure supporting their trading system.

![Infrastructure Monitoring](assets/infrastructure_monitoring.png)



### CI/CD DAG

The Continuous Integration/Continuous Deployment (CI/CD) pipeline is a crucial component of modern software development, especially in complex projects like our Deep Reinforcement Learning Stock Trading system. It automates the process of integrating code changes, testing, and deploying them to production, ensuring that the system is always running the latest, most stable version. A Directed Acyclic Graph (DAG) represents the sequence of operations or tasks in the CI/CD pipeline, illustrating the dependencies between tasks and how they are orchestrated.
Successful State

#### Successful State

In a successful state, the CI/CD DAG executes all tasks without errors, from code integration to deployment. This process begins with a developer pushing code changes to the repository, triggering the CI/CD pipeline. The pipeline then runs through several stages, such as linting, unit testing, integration testing, and deployment. Each task in the DAG is represented as a node, with edges indicating the flow and dependencies between tasks. A successful execution means that all tests pass, and the code changes are safely deployed to production, often visualized with green indicators in CI/CD tools.

![CI/CD Pipeline (successful)](assets/dag_successful.png)

#### Failed State

Conversely, a failed state in the CI/CD DAG occurs when one or more tasks in the pipeline fail. This could be due to a variety of reasons, such as syntax errors, failing tests, or deployment issues. When a failure occurs, the pipeline halts, preventing potentially unstable or broken code from being deployed. The failed task(s) are typically highlighted in red, and developers are notified to investigate and resolve the issue. The DAG visualization helps in quickly identifying the point of failure and understanding the dependencies that might be affected.

![CI/CD Pipeline (failure)](assets/ci_cd_dag_failure.png)

Importance of Monitoring CI/CD DAG

Monitoring the CI/CD DAG is essential for maintaining the health and efficiency of the development process. It provides real-time feedback on the state of the pipeline, enabling teams to quickly address issues and minimize downtime. Additionally, analyzing the patterns of success and failure over time can offer insights into common bottlenecks or areas for improvement in the codebase or the CI/CD process itself.

In summary, the CI/CD DAG is a powerful tool for visualizing and managing the flow of code from development to production. By closely monitoring its states, teams can ensure a smooth, efficient, and reliable development lifecycle for their Deep Reinforcement Learning Stock Trading system.



### Model Registry

The model registry process in the provided codebase is encapsulated within the RegisterModel step of the SageMaker pipeline. This step is responsible for registering the trained model into SageMaker's Model Registry, which allows for version control, model metadata storage, and model lifecycle management. The registration step is configured with various parameters such as content types, response types, inference instances, transform instances, and the model package group name. Additionally, it specifies the approval status of the model, which can be crucial for automated deployment processes in production environments.

Here's a brief overview of how the model registry step is set up in the pipeline:

```python
    step_register = RegisterModel(
        name="RegisterDLRModel",
        estimator=rl_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/x-parquet"],
        response_types=["application/x-parquet"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )

    step_cond = ConditionStep(
        name="DLREpisodeReturnCond",
        conditions=[cond_lte],
        if_steps=[
            step_register, 
            step_create_model, 
            step_deploy_endpoint
        ],
        else_steps=[],
    )
```

![Model Registry](assets/model_registry.png)


We adopt a unique approach to deployment that diverges from the conventional method of deploying models directly to a SageMaker endpoint. Instead, we leverage AWS Fargate, orchestrated by an AWS Lambda function, for deploying our trained models. This strategy allows us to benefit from the serverless computing environment provided by AWS Fargate, which offers both flexibility and scalability without the need to manage servers or clusters.

Here's how the deployment process works in our system:

1. Lambda Function Trigger: After the model has been registered in the SageMaker Model Registry with the necessary version control and metadata, a Lambda function is triggered. This function is responsible for initiating the deployment process to AWS Fargate.

2. Deployment to AWS Fargate: The Lambda function retrieves the model from the SageMaker Model Registry and deploys it to a container in AWS Fargate. AWS Fargate allows us to run containers without having to manage servers or clusters, providing a highly scalable and flexible environment for our trading model.

3. Configuration and Scaling: Within Fargate, we can easily configure the computing resources allocated to our model and scale up or down based on the demand. This is particularly useful for adapting to varying workloads in stock trading, where market conditions can change rapidly.

4. Endpoint Creation: Once deployed, the model runs within a container in Fargate, and an endpoint is created. This endpoint can be accessed by our trading system to make predictions and execute trades based on the model's insights.

5. Monitoring and Management: The entire process, from deployment to monitoring, is managed through AWS services, ensuring high availability and reliability. We can monitor the performance of our model in real-time and make adjustments as necessary.

By deploying to AWS Fargate using a Lambda function, we gain a more flexible and scalable deployment solution compared to traditional SageMaker endpoints. This approach aligns with our system's needs.


```python
    func = Lambda(
        function_name="DeployModelFunction",
        execution_role_arn=role,
        script=os.path.join(BASE_DIR, "lambda_helper.py"),
        handler="lambda_helper.endpoint_handler",
    )
    output_param_1 = LambdaOutput(
        output_name="statusCode", 
        output_type=LambdaOutputTypeEnum.String
    )
    output_param_2 = LambdaOutput(
        output_name="body", 
        output_type=LambdaOutputTypeEnum.String
    )
    step_deploy_endpoint = LambdaStep(
        name="EndpointCreateStep",
        lambda_func=func,
        inputs={
            "model_name": step_create_model.properties.ModelName,
            "endpoint_config_name": f"{pipeline_name}-endpoint-config",
            "endpoint_name": f"{pipeline_name}-endpoint",
            "execution_role_arn": role,
            "image_url": image_url)
        },
        outputs=[output_param_1, output_param_2],
    )
```


### Paper Trading Environment Deployment

Following the deployment of our Deep Reinforcement Learning (DRL) model to AWS Fargate, the next crucial step involves the model's interaction with a paper trading environment. Unlike traditional deployments where outputs might be straightforward predictions or classifications, the outputs in our scenario are the actions taken by the agent within the simulated trading environment. This approach allows us to evaluate the model's performance in a risk-free setting before considering real-world application.

#### Understanding Paper Trading Outputs

In the context of our DRL stock trading system, the outputs are not typical batch inference job results or endpoint invocation responses. Instead, the outputs are the trading actions executed by the DRL agent, such as buy, sell, or hold decisions for different stocks. These actions are based on the model's predictions and are executed within the paper trading environment, which simulates real market conditions without actual financial risk.

#### How It Works

1. Model Invocation: The deployed model in AWS Fargate is invoked periodically (e.g., every minute, hour, or day, depending on the trading strategy) by sending it the latest market data. This data includes stock prices, volumes, and possibly other financial indicators that the model was trained on.

2. Decision Making: Upon receiving the market data, the model processes it to make trading decisions. These decisions are determined by the policy the DRL agent has learned during its training phase, aiming to maximize the portfolio's value over time.

3. Action Execution: The trading actions decided by the model are then executed within the paper trading environment. This environment accurately reflects market conditions but uses simulated money, thus providing a safe platform for testing.

4. Performance Tracking: Each trade and its outcome are recorded to evaluate the model's trading strategy over time. Metrics such as return on investment (ROI), maximum drawdown, and Sharpe ratio can be calculated to assess performance.

5. Feedback Loop: The results of the paper trading can be used as feedback to further refine and improve the DRL model. This iterative process helps in optimizing the trading strategy and adjusting to new market conditions.

#### Benefits of Paper Trading Deployment

- Risk-Free Evaluation: Allows for the testing of trading strategies without the risk of losing real money.
- Realistic Market Interaction: Provides insights into how the model would perform under real market conditions.
- Strategy Refinement: Enables continuous improvement of the trading strategy based on performance metrics.
- Confidence Building: Builds confidence in the model's decision-making ability before any real-world application.

By deploying our DRL model to a paper trading environment, we gain valuable insights into its potential effectiveness and areas for improvement in stock trading strategies. This step is essential for validating the model's capabilities and ensuring its readiness for real-world financial markets.

<p align="center">
  <img src="assets/paper_trading.png" alt="Agentic Trading Within Simulated Environment">
</p>
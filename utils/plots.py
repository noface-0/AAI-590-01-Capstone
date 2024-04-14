import os
import json
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)

def plot_drl_learning(
        path: str,
        eval_path: str,
        agent: str='ppo'
):
    title_dict = {
        "ppo": 'Proximal Policy Optimization (PPO)',
        "sac": 'Soft-Actor Critic (SAC)'
    }
    title = title_dict.get(agent)

    data = pd.read_csv(path)

    with open(eval_path, 'r') as file:
        eval_data = json.load(file)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Total Step')
    ax1.set_ylabel('Objective C', color=color)
    ax1.plot(
        data['Total Step'], 
        data['Objective C'], 
        label='Objective C', color=color
    )

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Objective A', color=color)
    ax2.plot(
        data['Total Step'], 
        data['Objective A'], 
        label='Objective A', 
        color=color
    )

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f'{title} Objective Values Over Steps')

    metrics_text = '\n'.join([
        f"Final Episode Return: {eval_data['final_episode_return']:.3f}",
        f"Max Drawdown: {eval_data['max_drawdown']:.3f}",
        f"Sharpe Ratio: {eval_data['sharpe_ratio']:.3f}"
    ])
    fig.text(
        1.02, 1.0, 
        metrics_text, 
        fontsize=9, 
        verticalalignment='top', 
        horizontalalignment='left', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    save_name = agent.lower() \
        .replace(" ", "_").replace("(", "_").replace(")", "_")

    plt.savefig(
        f'{BASE_DIR}/assets/{save_name}_learning_curve.png', 
        bbox_inches="tight"
    )
    plt.show()


def plot_nn_learning(
        path: str,
        eval_path: str,
        model_name: str='Feedforward Neural Network (FNN)'
):
    with open(path, 'r') as file:
        loss_data = json.load(file)

    with open(eval_path, 'r') as file:
        eval_data = json.load(file)
    
    training_losses = loss_data['training_losses']
    validation_losses = loss_data['validation_losses']
    steps = list(range(1, len(training_losses) + 1))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:orange'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(
        steps, 
        training_losses, 
        label='Training Loss', 
        color=color
    )
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Loss', color=color)
    ax2.plot(
        steps, 
        validation_losses, 
        label='Validation Loss', 
        color=color
    )
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'{model_name} Loss Over Epochs')
    
    metrics_text = '\n'.join([
        f"RMSE: {eval_data['RMSE']:.3f}",
        f"R-squared: {eval_data['R-squared']:.3f}"
    ])
    fig.text(
        1.02, 1.0, 
        metrics_text, 
        fontsize=9, 
        verticalalignment='top', 
        horizontalalignment='left', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    save_name = model_name.lower() \
        .replace(" ", "_").replace("(", "_").replace(")", "_")
    
    plt.savefig(
        f'{BASE_DIR}/assets/{save_name}_learning_curve.png', 
        bbox_inches="tight"
    )
    plt.show()
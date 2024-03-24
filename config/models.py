# Model Parameters
ERL_PARAMS={
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64, 
    "gamma":  0.985, 
    "ent_coef": 0.01,
    "seed":312, 
    "net_dimension":[256, 256], 
    "target_step":5000, 
    "eval_gap":30, 
    "eval_times":1
}

A2C_PARAMS = {
    "n_steps": 5, 
    "ent_coef": 0.01, 
    "learning_rate": 0.0007
}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}

DDPG_PARAMS = {
    "batch_size": 128, 
    "buffer_size": 50000, 
    "learning_rate": 0.001
}

TD3_PARAMS = {
    "batch_size": 100, 
    "buffer_size": 1000000, 
    "learning_rate": 0.001
}

SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
    "net_dimension": [256, 256]
}

GA_PARAMS = {
    "mutation_rate": 0.1,
    "num_generations": 100,
}
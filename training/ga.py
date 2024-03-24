import json
import pandas as pd

from models.ga import GeneticAlgorithm


def evolve_portfolio(
    objective: str,
    num_generations: int,
    mutation_rate: float,
    start_date,
    end_date,
    ticker_list,
    time_interval,
    data=pd.DataFrame(),
    **kwargs,
):
    population_size = len(ticker_list)

    ga = GeneticAlgorithm(
        objective=objective,
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        start_date=start_date,
        end_date=end_date,
        time_interval=time_interval,
        ticker_list=ticker_list,
        data=data
    )

    best_individual, best_returns, best_drawdown = ga.run()

    best_individual_data = {
        "best_individual": best_individual.tolist(),
        "returns": best_returns,
        "drawdown": best_drawdown
    }
    with open(f'models/runs/ga/{objective}.json', 'w') as json_file:
        json.dump(best_individual_data, json_file, indent=4)

    print("Best individual (tickers):", best_individual)
    print("Returns:", best_returns)
    print("Drawdown:", best_drawdown)

    return best_individual
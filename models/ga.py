import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from processing.extract import download_data
from utils.utils import get_var

API_KEY = get_var("ALPACA_API_KEY")
API_SECRET = get_var("ALPACA_API_SECRET")
API_BASE_URL = get_var("ALPACA_API_BASE_URL")


class GeneticAlgorithm:
    def __init__(
            self,
            objective,
            population_size, 
            num_generations, 
            mutation_rate,
            time_interval,
            start_date,
            end_date,
            ticker_list,
            data=pd.DataFrame(),
            **kwargs,
    ):
        self.objective = objective
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.ticker_list = ticker_list
        self.kwargs = kwargs

        self.best_individual = None
        self.returns = None
        self.drawdown = None

    def create_individual(self):
        tickers = np.random.choice(
            self.ticker_list, 
            size=random.randint(1, 10), 
            replace=False
        )
        return tickers

    def create_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        return population

    def simulate(self, individual):
        selected_data = self.data[self.data['tic'].isin(individual)]

        if not selected_data.empty:
            ohlc_data = selected_data[['open', 'high', 'low', 'close']]
            
            daily_returns = ohlc_data['close'].pct_change()
            
            cumulative_returns = (1 + daily_returns).cumprod()
            max_upside = (cumulative_returns.max() - 1) * 100
            max_drawdown = (cumulative_returns.min() - 1) * 100
            
            return max_upside, max_drawdown
        else:
            return 0, 0

    def fitness(self, individual):
        returns, drawdown = self.simulate(individual)
        
        if self.objective == 'max_returns':
            fitness = returns
        elif self.objective == 'min_drawdown':
            fitness = -drawdown
        else:  # 'max_returns_min_drawdown'
            fitness = returns / drawdown
        
        return fitness

    def selection(self, population):
        # Perform tournament selection to select parents
        selected_parents = []
        for _ in range(len(population)):
            tournament = random.sample(population, k=20)
            best_individual = max(tournament, key=self.fitness)
            selected_parents.append(best_individual)
        return selected_parents

    def crossover(self, parent1, parent2):
        # Single-point crossover for offspring creation
        cross_pt = random.randint(1, min(len(parent1), len(parent2)) - 1)
        offspring1 = np.concatenate((parent1[:cross_pt], 
                                     parent2[cross_pt:]))
        offspring2 = np.concatenate((parent2[:cross_pt], 
                                     parent1[cross_pt:]))
        return offspring1, offspring2

    def mutation(self, individual):
        mutated_individual = individual.copy()
        if random.random() < self.mutation_rate:
            new_ticker = np.random.choice(self.ticker_list, size=1)[0]
            index = random.randint(0, len(mutated_individual) - 1)
            mutated_individual[index] = new_ticker
        return mutated_individual

    def run(self):
        if self.data.empty:
            self.data = download_data(
                ticker_list=self.ticker_list,
                start_date=self.start_date,
                end_date=self.end_date,
                time_interval=self.time_interval,
                api_key=API_KEY,
                api_secret=API_SECRET,
                api_base_url=API_BASE_URL
            )

        objective = self.objective
        population = self.create_population()
        
        for generation in tqdm(
            range(self.num_generations), desc="Evolving Generations"
        ):
            parents = self.selection(population)
            offspring = []
            
            for i in range(0, len(parents) - len(parents) % 2, 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([child1, child2])
            
            offspring = [self.mutation(child) for child in offspring]
            population = offspring
            
            # evaluate individuals
            for individual in population:
                returns, drawdown = self.simulate(individual)
                if (
                    self.returns is None
                    or (objective == 'max_returns' and returns > self.returns)
                    or (objective == 'min_drawdown' and drawdown < self.drawdown)
                ):
                    self.best_individual = individual
                    self.returns = returns
                    self.drawdown = drawdown
        
        return self.best_individual, self.returns, self.drawdown
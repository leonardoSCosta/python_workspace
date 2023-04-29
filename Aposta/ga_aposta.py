"""
Aposta resolvia por GA
"""
import random
import numpy as np
from copy import deepcopy

N_BETS = 3
BIT_MASK = 0xFFFFFF
MONEY = 10

# ODDS
BET_A = np.asarray([-1, -1, 2.41])
BET_B = np.asarray([3, -1, -1])
BET_C = np.asarray([-1,  2, -1])
BET_B_C = np.asarray([3,  2, -1])

SOL = MONEY * np.asarray([28, 38, 33])/100
print(np.sum(SOL))
print(SOL)

print(np.dot(SOL, BET_A) - MONEY, np.dot(SOL, BET_B) - MONEY,
      np.dot(SOL, BET_C) - MONEY, np.dot(SOL, BET_B_C) - MONEY)


class Chromossome():
    """
    Representa um cromossomo
    """

    def __init__(self):
        self.g_code = np.uint32(0)
        self.set_val(random.randint(0, 100),
                     random.randint(0, 100),
                     random.randint(0, 100))

    @staticmethod
    def crossover(ind_a, ind_b):
        cross_point = random.randint(0, 8*N_BETS)
        mask = int(2**cross_point - 1) & BIT_MASK
        not_mask = (~mask) & BIT_MASK

        child_a = np.uint32(0)
        child_b = np.uint32(0)
        child_a = ind_a.g_code & mask | ind_b.g_code & not_mask
        child_b = ind_b.g_code & mask | ind_a.g_code & not_mask

        ind_a.g_code = child_a
        ind_b.g_code = child_b

        val = ind_a.get_val()
        ind_a.set_val(val[0], val[1], val[2])
        val = ind_b.get_val()
        ind_b.set_val(val[0], val[1], val[2])

    def mutate(self):
        mut_point = random.randint(0, 8*N_BETS)
        mask = int(2**mut_point) & BIT_MASK

        self.g_code = self.g_code & ~(self.g_code & mask)

    def get_val(self):
        values = [0, 0, 0]
        for i in range(0, N_BETS):
            values[i] = 0xFF & (self.g_code >> int(i*8))
        return values

    def set_val(self, bet_a, bet_b, bet_c):
        total = bet_a + bet_b + bet_c + 1e-10
        values = [round(100*bet_a/total),
                  round(100*bet_b/total),
                  round(100*bet_c/total)]

        total = sum(values)

        self.g_code = np.uint32(0)
        for i in range(0, N_BETS):
            self.g_code = self.g_code | (values[i] << int(i*8))

        if total < 100:
            self.set_val(random.randint(0, 100),
                         random.randint(0, 100),
                         random.randint(0, 100))


class GA():
    """
    Otimização por GA
    """

    def __init__(self, prob_cross: float, prob_mut: float, n_indv: int,
                 money: int):
        self.PROB_CROSSOVER = prob_cross
        self.PROB_MUTATION = prob_mut
        self.MONEY = money

        if n_indv % 2 != 0:
            n_indv += 1
        self.N = n_indv
        self.population = []

        for _ in range(self.N):
            self.population.append(Chromossome())

    def crossover(self):
        for i in range(0, self.N, 2):
            if random.randint(0, 100)/100 < self.PROB_CROSSOVER:
                Chromossome.crossover(self.population[i], self.population[i+1])

    def mutate(self):
        for ind in self.population:
            if random.randint(0, 100)/100 < self.PROB_MUTATION:
                ind.mutate()

    @staticmethod
    def roulette_wheel(sum_fitness) -> int:
        max_v = np.max(sum_fitness)
        min_v = np.min(sum_fitness)
        rng_gen = np.random.default_rng()
        random_val = min_v + abs(max_v - min_v) * rng_gen.random(1)

        for i, fit_val in enumerate(sum_fitness):
            if fit_val >= random_val:
                return i
        return 0

    def get_fitness(self, ind: Chromossome):
        values = self.MONEY * np.asarray(ind.get_val()) / 100.0

        G1 = np.dot(BET_A, values) - self.MONEY
        G2 = np.dot(BET_B, values) - self.MONEY
        G3 = np.dot(BET_C, values) - self.MONEY
        G4 = np.dot(BET_B_C, values) - self.MONEY

        return min(G1, G2, G3, G4)

    def selection(self, best_ind: Chromossome):
        selected_index = self.N * [0]
        sum_fitness = self.N * [0]

        for i in range(self.N):
            if i >= 1:
                sum_fitness[i] = self.get_fitness(self.population[i]) + \
                    sum_fitness[i-1]
            else:
                sum_fitness[i] = self.get_fitness(self.population[i])

        for i in range(0, self.N):
            selected_index[i] = self.roulette_wheel(sum_fitness)

        selected_pop = self.population.copy()
        for i, index in enumerate(selected_index):
            selected_pop[i] = self.population[index]

        self.population.clear()
        self.population.extend(selected_pop)
        # Elitismo
        self.population[0] = best_ind

    def best_solution(self):
        best_i = 0
        best_fit = self.get_fitness(self.population[0])
        for n, ind in enumerate(self.population):
            current_fit = self.get_fitness(ind)
            if current_fit > best_fit:
                best_fit = current_fit
                best_i = n
        return deepcopy(self.population[best_i])

    def run(self):
        MAX_I = 2000
        best_ind = Chromossome()
        for _ in range(MAX_I):
            prev_best = best_ind
            best_ind = self.best_solution()
            if self.get_fitness(best_ind) != self.get_fitness(prev_best):
                print("Melhor Solucao: ", best_ind.get_val(),
                      self.get_fitness(best_ind))

            self.crossover()
            self.mutate()
            self.selection(best_ind)
        print("**************** RESULTADO ****************")
        print(best_ind.get_val())


if __name__ == "__main__":
    ga_solv = GA(0.9, 0.05, 200, MONEY)
    ga_solv.run()

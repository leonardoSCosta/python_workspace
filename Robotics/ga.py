"""
Aposta resolvia por GA
"""
from copy import deepcopy, copy
import random
import numpy as np

import cubic_spline_planner
import lqr_speed_steer_control

#  Q_SIZE = 4 * 4
#  Q_INDEX = 4
#  R_SIZE = 3 * 3
#  R_INDEX = 3

Q_SIZE = 7 * 7
Q_INDEX = 7
R_SIZE = 3 * 3
R_INDEX = 3

#  X_COORD = [0.0, 1.0]
#  Y_COORD = [5.0, 8.0]
X_COORD = [10.0, 14.0, 16.0, 22.5, 20.0, 27.5, 30.0, 35.0, 40.0]
Y_COORD = [-5.0, 0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0, 10.0]
GOAL = [X_COORD[-1], Y_COORD[-1]]
TARGET_SPEED = 2.0

COURSE_X, COURSE_Y, COURSE_W, COURSE_K, COURSE_S = \
    cubic_spline_planner.calc_spline_course(X_COORD, Y_COORD, ds=0.1)

SPEED_PROFILE = lqr_speed_steer_control.calc_speed_profile(COURSE_W,
                                                           TARGET_SPEED)
RESET_VALUE = 99999999999


class Chromossome():
    """
    Representa um cromossomo
    """
    def __init__(self):
        # self.g_code = np.random.random((1, Q_SIZE + R_SIZE))
        self.fitness = RESET_VALUE
        self.g_code = np.asarray(
                [[-22.19186988, 0.92622235, 13.17825549, -11.90768075,
                 0.88223625, 23.91207102, 0.79587746, 0.35485784,
                 26.0648412, 6.50328178, 0.68100182, 0.11575109,
                 0.95873849, 0.82054465, 0.69532375, -22.77513172,
                 -6.68813916, 0.81403336, 0.07317548,
                 0.34722787, 0.375804, 0.49415276,
                 0.73610013, 0.35538129, 16.49761124]])

    @staticmethod
    def crossover(ind_a, ind_b):
        cross_point = random.randint(0, Q_SIZE + R_SIZE-1)

        offspring_a = np.zeros((1, Q_SIZE + R_SIZE))
        offspring_b = np.zeros((1, Q_SIZE + R_SIZE))

        offspring_a[0, 0:cross_point] = copy(ind_a.g_code[0, 0:cross_point])
        offspring_a[0, cross_point:] = copy(ind_b.g_code[0, cross_point:])

        offspring_b[0, 0:cross_point] = copy(ind_b.g_code[0, 0:cross_point])
        offspring_b[0, cross_point:] = copy(ind_a.g_code[0, cross_point:])

        ind_a.g_code = offspring_a
        ind_b.g_code = offspring_b

    def mutate(self):
        mut_point = random.randint(0, Q_SIZE + R_SIZE-1)
        self.g_code[0, mut_point] = random.uniform(-30, 30)

    def get_Q(self):
        return self.g_code[0, 0:Q_SIZE].reshape((Q_INDEX, Q_INDEX))

    def get_R(self):
        return self.g_code[0, Q_SIZE:].reshape((R_INDEX, R_INDEX))


class GA():
    """
    Otimização por GA
    """
    def __init__(self, prob_cross: float, prob_mut: float, n_indv: int):
        self.PROB_CROSSOVER = prob_cross
        self.PROB_MUTATION = prob_mut

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
        if ind.fitness >= RESET_VALUE:
            lqr_speed_steer_control.lqr_Q = ind.get_Q()
            lqr_speed_steer_control.lqr_R = ind.get_R()

            t, x, y, w, _, _, _, _, theta, e = \
                lqr_speed_steer_control.do_simulation(COURSE_X, COURSE_Y,
                                                      COURSE_W, COURSE_K,
                                                      SPEED_PROFILE, GOAL)
            dist = [value**2 for value in e]
            angle = [value**2 for value in theta]
            fitness = sum(dist)*5 + sum(angle) + t[-1] * 2

            if t[-1] > 41.9:
                fitness += 1000

            ind.fitness = 1.0/fitness
        return ind.fitness

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

    def reset_fitness(self):
        for ind in self.population:
            ind.fitness = RESET_VALUE

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
        MAX_I = 1000
        best_ind = Chromossome()
        for i in range(MAX_I):
            print("Iteration: ", i)
            prev_best = best_ind
            best_ind = self.best_solution()
            if self.get_fitness(best_ind) != self.get_fitness(prev_best):
                print("Melhor Solucao: \nQ:\n", best_ind.get_Q(),
                      "\nR:\n", best_ind.get_R(),
                      "\nFitness: ", self.get_fitness(best_ind))

            self.crossover()
            self.mutate()
            self.selection(best_ind)
            self.reset_fitness()
        print("**************** RESULTADO ****************")
        print("Q:\n", best_ind.get_Q(), "\nR:\n", best_ind.get_R())
        lqr_speed_steer_control.show_animation = True
        self.get_fitness(best_ind)


if __name__ == "__main__":
    ga_solv = GA(0.9, 0.05, 100)
    ga_solv.run()

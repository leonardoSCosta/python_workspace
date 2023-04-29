"""
Otimização matrizes Q e R do LQR
"""
from copy import deepcopy, copy
from math import sqrt
from multiprocessing import Pool
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt

import cubic_spline_planner
#  import lqr_speed_steer_control
import lqr_speed_steer_control_edit as lqr_speed_steer_control

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

DEBUG = False
RANDOM_INIT = False


class Chromossome():
    """
    Representa um cromossomo
    """
    id = 0

    def __init__(self):
        self.fitness = RESET_VALUE
        self.fitness_calculated = False
        self.c_id = Chromossome.id

        if DEBUG:
            print("Criando cromossomo", self.c_id)
        Chromossome.id = Chromossome.id + 1

        if not RANDOM_INIT:
            # matriz_Q = np.ravel(np.eye(7))
            # matriz_R = np.ravel(np.eye(3))
            matriz_Q = np.ravel(np.loadtxt("./matriz_Q_boa_tremendo.txt"))
            matriz_R = np.ravel(np.loadtxt("./matriz_R_boa_tremendo.txt"))
            self.g_code = np.concatenate((matriz_Q, matriz_R), axis=0)
            # self.g_code = np.ones(58)
        else:
            self.g_code = np.random.random(Q_SIZE + R_SIZE)

    @staticmethod
    def crossover(ind_a, ind_b):
        cross_point = random.randint(0, Q_SIZE + R_SIZE-1)

        offspring_a = np.zeros(Q_SIZE + R_SIZE)
        offspring_b = np.zeros(Q_SIZE + R_SIZE)

        offspring_a[0:cross_point] = copy(ind_a.g_code[0:cross_point])
        offspring_a[cross_point:] = copy(ind_b.g_code[cross_point:])

        offspring_b[0:cross_point] = copy(ind_b.g_code[0:cross_point])
        offspring_b[cross_point:] = copy(ind_a.g_code[cross_point:])

        ind_a.g_code = offspring_a
        ind_b.g_code = offspring_b

    def mutate(self):
        mut_point = random.randint(0, Q_SIZE + R_SIZE-1)
        self.g_code[mut_point] = random.uniform(-30, 30)

    def get_Q(self):
        return self.g_code[0:Q_SIZE].reshape((Q_INDEX, Q_INDEX))

    def get_R(self):
        return self.g_code[Q_SIZE:].reshape((R_INDEX, R_INDEX))


def calc_fitness(ind: Chromossome):
    if DEBUG:
        print("Calculando a aptidão do cromossomo ", ind.c_id)

    ind.fitness_calculated = True
    lqr_speed_steer_control.lqr_Q = ind.get_Q()
    lqr_speed_steer_control.lqr_R = ind.get_R()

    t, x, y, w, _, _, _, _, theta, e, e_dot, theta_dot, e_dot_dot, theta_dot_dot = \
        lqr_speed_steer_control.do_simulation(COURSE_X, COURSE_Y,
                                              COURSE_W, COURSE_K,
                                              SPEED_PROFILE, GOAL)
    dist = [value**2 for value in e]
    angle = [value**2 for value in theta]
    e_derivate = [value**2 for value in e_dot]
    theta_derivate = [value**2 for value in theta_dot]
    e_derivate_derivate = [value**2 for value in e_dot_dot]
    theta_derivate_derivate = [value**2 for value in theta_dot_dot]

    dist_value = sum(dist)*100/5e7
    angle_value = sum(angle)*100/8e4
    time_value = t[-1]/42
    e_dot_value = sum(e_derivate)/1e10
    e_ddot_value = sum(e_derivate_derivate)/5e12
    theta_dot_value = sum(theta_derivate)/5e7
    theta_ddot_value = sum(theta_derivate_derivate)/1e10

    fitness = dist_value + angle_value + time_value + e_dot_value + \
        theta_dot_value + e_ddot_value + theta_ddot_value

    if t[-1] > 41.9:
        fitness += 4000
    fitness = 1.0/fitness

    return fitness, time_value, dist_value, angle_value, e_dot_value, \
        e_ddot_value, theta_dot_value, theta_ddot_value, ind.c_id


class GA():
    """
    Otimização por GA
    """

    def __init__(self, prob_cross: float, prob_mut: float, n_indv: int,
                 n_iterations: int):
        self.PROB_CROSSOVER = prob_cross
        self.PROB_MUTATION = prob_mut

        self.MAX_I = n_iterations
        self.current_it = 0

        if n_indv % 2 != 0:
            n_indv += 1
        self.N = n_indv
        self.population = []

        self.fitness_list = []
        self.time_list = []
        self.e_list = []
        self.theta_list = []
        self.e_dot_list = []
        self.e_ddot_list = []
        self.theta_dot_list = []
        self.theta_ddot_list = []

        self.fitness_max = -1
        self.time_max = -1
        self.e_max = -1
        self.e_dot_max = -1
        self.e_ddot_max = -1
        self.theta_max = -1
        self.theta_dot_max = -1
        self.theta_ddot_max = -1

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

    def verify_max(self, value_list):
        max_value = max(value_list)
#          for values in value_list:
#              if values >= max_value:
#                  max_value = values
#              elif values < max_value:
#                  max_value = max_value
        return max_value

    def get_fitness(self, ind: Chromossome):
        if ind.fitness_calculated is False:
            print("Lendo fitness sem calcular.... : Cromossomo", ind.c_id)
            return RESET_VALUE
        return ind.fitness

    def set_fitness(self, ind: Chromossome, fitness):
        if ind.fitness_calculated is False:
            ind.fitness_calculated = True
            ind.fitness = fitness

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

        selected_pop = deepcopy(self.population)
        for i, index in enumerate(selected_index):
            selected_pop[i] = deepcopy(self.population[index])

        self.population.clear()
        self.population.extend(selected_pop)
        # Elitismo
        self.population[0] = best_ind

    def reset_fitness(self):
        for n, ind in enumerate(self.population):
            ind.fitness_calculated = False
            ind.c_id = n

    def best_solution(self):
        best_i = 0
        best_fit = -1

        for n, ind in enumerate(self.population):

            if self.get_fitness(ind) > best_fit:
                best_fit = self.get_fitness(ind)
                best_i = n

        return deepcopy(self.population[best_i])

    def update_fitness(self):
        start_time = time()
        print("Avaliando a aptidão dos individuos...")
        with Pool() as p:
            val = p.imap(calc_fitness, self.population)
            for result in val:
                fitness = result[0]
                dt = result[1]
                e = result[2]
                theta = result[3]
                e_dot = result[4]
                e_ddot = result[5]
                theta_dot = result[6]
                theta_ddot = result[7]
                ind_id = result[8]

                self.set_fitness(self.population[ind_id], fitness)

                self.fitness_list.append(fitness)
                self.time_list.append(dt)
                self.e_list.append(e)
                self.theta_list.append(theta)
                self.e_dot_list.append(e_dot)
                self.e_ddot_list.append(e_ddot)
                self.theta_dot_list.append(theta_dot)
                self.theta_ddot_list.append(theta_ddot)

                self.fitness_max = self.verify_max(self.fitness_list)
                self.time_max = self.verify_max(self.time_list)
                self.e_max = self.verify_max(self.e_list)
                self.e_dot_max = self.verify_max(self.e_dot_list)
                self.e_ddot_max = self.verify_max(self.e_ddot_list)
                self.theta_max = self.verify_max(self.theta_list)
                self.theta_dot_max = self.verify_max(self.theta_dot_list)
                self.theta_ddot_max = self.verify_max(self.theta_ddot_list)
        total_time = round(time() - start_time, 3)
        ind_per_s = round(self.N / total_time)
        eta = (self.MAX_I - self.current_it) * self.N / ind_per_s
        print("Levou", total_time, "segundos ---",
              round(ind_per_s), "ind/s --- ETA =",
              round(eta), "segundos =", round(eta/60), "minutos =",
              round(eta/3600), "horas")

    def run(self):
        best_ind = self.population[0]
        for self.current_it in range(self.MAX_I):
            print("\nIteration: ", self.current_it)
            print("\n")
            self.update_fitness()

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

        print(f"Max value fitness: {self.fitness_max:.2e}")
        print(f"Max value time: {self.time_max:.2e}")
        print(f"Max value distance: {self.e_max:.2e}")
        print(f"Max value first derivate distance: {self.e_dot_max:.2e}")
        print(f"Max value second derivate distance: {self.e_ddot_max:.2e}")
        print(f"Max value theta: {self.theta_max:.2e}")
        print(f"Max value first derivate theta: {self.theta_dot_max:.2e}")
        print(f"Max value second derivate theta: {self.theta_ddot_max:.2e}")

        plt.figure()
        # plt.plot(self.fitness_list, "r", label='Fitness values')
        # plt.plot(self.e_list, "g", label='Distance values')
        # plt.plot(self.e_dot_list, "b", label='Distance first derivate values')
        # plt.plot(self.e_ddot_list, "c", label='Distance second derivate values')
        # plt.plot(self.theta_list, "m", label='Theta values')
        # plt.plot(self.theta_dot_list, "y", label='Theta first derivate values')
        # plt.plot(self.theta_ddot_list, "k", label='Theta second derivate values')
        plt.plot(self.time_list, '-c', label="Time values")
        length_list = len(self.fitness_list)
        plt.figure()
        plt.stackplot(range(0, length_list), self.fitness_list, self.e_list, self.e_dot_list, self.e_ddot_list, self.theta_list, self.theta_dot_list, self.theta_ddot_list, self.time_list, labels = ['Fitness', 'Distance', 'Distance_dot', 'Distance_ddot', 'Theta', 'Theta_dot', 'Theta_ddot', 'Time'], colors=['r', 'g', 'b', 'c', 'm', 'y', 'k', 'dimgray'])
        plt.legend()
        plt.show()

        np.savetxt("arquivo_matriz_Q_5.txt", best_ind.get_Q())
        np.savetxt("arquivo_matriz_R_5.txt", best_ind.get_R())

        lqr_speed_steer_control.show_animation = True
        best_ind.fitness = RESET_VALUE
        best_ind.fitness_calculated = False
        calc_fitness(best_ind)


if __name__ == "__main__":
    ga_solv = GA(prob_cross=0.9, prob_mut=0.05, n_indv=50, n_iterations=100)
    ga_solv.run()

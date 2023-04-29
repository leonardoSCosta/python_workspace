#!/usr/bin/python3.8
"""
Otimização matrizes Q e R do LQR
"""
from copy import deepcopy, copy
from math import radians, degrees, pi, floor
from multiprocessing import Pool
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt

# import cubic_spline_planner
#  import ssl_simulation_2 as simulator
import ssl_simulation_wheel as simulator
#  import lqr_speed_steer_control_edit as lqr_speed_steer_control

#  X_COORD = [0.0, 1.0]
#  Y_COORD = [5.0, 8.0]
# X_COORD = [10.0, 14.0, 16.0, 22.5, 20.0, 27.5, 30.0, 35.0, 40.0]
# Y_COORD = [-5.0, 0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0, 10.0]
# GOAL = [X_COORD[-1], Y_COORD[-1]]
# TARGET_SPEED = 2.0

RESET_VALUE = 99999999999

DEBUG = False


def rpm_to_rads(rpm):
    return rpm * 2 * pi / 60


class Chromossome():
    """
    Representa um cromossomo
    """
    id = 0

    def __init__(self):
        self.g_code = []
        self.fitness = RESET_VALUE
        self.fitness_calculated = False
        self.c_id = Chromossome.id
        if DEBUG:
            print("Criando cromossomo", self.c_id)
        Chromossome.id = Chromossome.id + 1
        angle_1 = format(np.random.randint(5, 30), "06b")
        angle_2 = format(np.random.randint(5, 30), "06b")
        for algarismos in angle_1:
            self.g_code.append(int(algarismos))
        for algarismos in angle_2:
            self.g_code.append(int(algarismos))

    @staticmethod
    def crossover(ind_a, ind_b):
        cross_point = random.randint(0, 5)

        offspring_a = np.zeros(12)
        offspring_b = np.zeros(12)

        offspring_a[0:cross_point] = copy(ind_a.g_code[0:cross_point])
        offspring_a[cross_point:6] = copy(ind_b.g_code[cross_point:6])
        offspring_a[6:(6+cross_point)] = copy(ind_a.g_code[6:(6+cross_point)])
        offspring_a[(6+cross_point):] = copy(ind_b.g_code[(6+cross_point):])

        offspring_b[0:cross_point] = copy(ind_b.g_code[0:cross_point])
        offspring_b[cross_point:6] = copy(ind_a.g_code[cross_point:6])
        offspring_b[6:(6+cross_point)] = copy(ind_b.g_code[6:(6+cross_point)])
        offspring_b[(6+cross_point):] = copy(ind_a.g_code[(6+cross_point):])

        ind_a.g_code = offspring_a
        ind_b.g_code = offspring_b

#          binary_a_alpha = ''.join(str(int(bit)) for bit in ind_a.g_code[0:6])
#          binary_b = ''.join(str(int(bit)) for bit in ind_b.g_code[0:6])

        ind_a_lower_value = int(''.join(str(int(bit))
                                        for bit in ind_a.g_code[0:6]), 2)
        ind_a_upper_value = int(''.join(str(int(bit))
                                        for bit in ind_a.g_code[6:12]), 2)

        ind_b_lower_value = int(''.join(str(int(bit))
                                        for bit in ind_b.g_code[0:6]), 2)
        ind_b_upper_value = int(''.join(str(int(bit))
                                        for bit in ind_b.g_code[6:12]), 2)

        if ind_a_lower_value < 5:
            ind_a.g_code[0:6] = [0, 0, 0, 1, 0, 1]
        elif ind_a_lower_value > 30:
            ind_a.g_code[0:6] = [0, 1, 1, 1, 1, 0]

        if ind_a_upper_value < 5:
            ind_a.g_code[6:12] = [0, 0, 0, 1, 0, 1]
        elif ind_a_upper_value > 30:
            ind_a.g_code[6:12] = [0, 1, 1, 1, 1, 0]

        if ind_b_lower_value < 5:
            ind_b.g_code[0:6] = [0, 0, 0, 1, 0, 1]
        elif ind_b_lower_value > 30:
            ind_b.g_code[0:6] = [0, 1, 1, 1, 1, 0]

        if ind_b_upper_value < 5:
            ind_b.g_code[6:12] = [0, 0, 0, 1, 0, 1]
        elif ind_b_upper_value > 30:
            ind_b.g_code[6:12] = [0, 1, 1, 1, 1, 0]

    def mutate(self):
        mut_point = random.randint(0, 11)
        self.g_code[mut_point] = random.randint(0, 1)

    def get_alpha(self):
        binary = ''.join(str(int(bit)) for bit in self.g_code[0:6])
        if binary == '000000':
            self.g_code[0:6] = [0, 0, 0, 0, 0, 1]
            binary = ''.join(str(int(bit)) for bit in self.g_code[0:6])
        decimal = int(binary, 2)
        if decimal == 0:
            print('Resultado inválido, alpha')
        return radians(decimal)

    def get_beta(self):
        binary = ''.join(str(int(bit)) for bit in self.g_code[6:])
        if binary == '000000':
            # print("Binário zerado")
            self.g_code[6:] = [0, 0, 0, 0, 0, 1]
            binary = ''.join(str(int(bit)) for bit in self.g_code[6:])
        decimal = int(binary, 2)
        if decimal == 0:
            print('Resultado inválido, beta')
        return radians(decimal)


def calc_fitness(ind: Chromossome):
    if DEBUG:
        print("Calculando a aptidão do cromossomo ", ind.c_id)

    ind.fitness_calculated = True

    velocity = 2.5
    goal_position = [[10], [0], [0]]

    alpha_value = ind.get_alpha()
    beta_value = ind.get_beta()

    w_1, w_2, w_3, w_4 = simulator.do_simulation(velocity, alpha_value,
                                                 beta_value, goal_position)

    w1_value = abs(w_1) / 100
    w2_value = abs(w_2) / 100
    w3_value = abs(w_3) / 100
    w4_value = abs(w_4) / 100

    alpha_value = ind.get_alpha()
    beta_value = ind.get_beta()
    # time_value = sum(time)*5/1e2

    fitness = 1  # w1_value + w2_value + w3_value + w4_value

    alpha_deg = degrees(alpha_value)
    beta_deg = degrees(beta_value)
    if alpha_deg > 80 or beta_deg > 80:
        # print('Valor incompatível de ângulo, alpha = {}, beta = {}. Fitness +
        # 1000000'.format(alpha_deg, beta_deg))
        fitness += 1e10

    if alpha_deg < 20 or beta_deg < 20:
        # print('Valor incompatível de ângulo, alpha = {}, beta = {}. Fitness +
        # 1000000'.format(alpha_deg, beta_deg))
        fitness += 1e10
    fitness = 1.0/fitness

    w_out_required = rpm_to_rads(3000/3)

    # Calcula as componentes para o robô andar em X
    kin, inv_kin = simulator.compute_matrixes(alpha_value, beta_value)
    w_out = simulator.compute_inverse_kinematic(np.asarray([[1], [0.5], [0]]),
                                                kin)
    # Valor máximo de velocidade angular para o robô andar em X
    max_w_in = w_out_required * w_out / np.linalg.norm(w_out)

    # Velocidade máxima em X com os ângulos alpha e beta
    velocidade = simulator.compute_kinematics(max_w_in, inv_kin)
#      print(degrees(alpha_value), degrees(beta_value), np.transpose(velocidade))
#      input()
    velocidade_x = velocidade[0][0] + velocidade[1][0]
    fitness += velocidade_x

    return fitness, w1_value, w2_value, w3_value, w4_value,\
        velocidade_x, ind.c_id


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
        self.w1_list = []
        self.w2_list = []
        self.w3_list = []
        self.w4_list = []
        self.vel_list = []

        self.fitness_max = -1
        self.time_max = -1
        self.w1_max = -1
        self.w2_max = -1
        self.w3_max = -1
        self.w4_max = -1
        self.vel_max = -1

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

    def update_population(self, result):
        fitness = result[0]
        w1 = result[1]
        w2 = result[2]
        w3 = result[3]
        w4 = result[4]
        velocidade = result[5]
        ind_id = result[6]

        self.set_fitness(self.population[ind_id], fitness)

        self.fitness_list.append(fitness)
        self.w1_list.append(1/w1)
        self.w2_list.append(1/w2)
        self.w3_list.append(1/w3)
        self.w4_list.append(1/w4)
        self.vel_list.append(velocidade)

#          self.fitness_max = self.verify_max(self.fitness_list)
#          self.w1_max = self.verify_max(self.w1_list)
#          self.w2_max = self.verify_max(self.w2_list)
#          self.w3_max = self.verify_max(self.w3_list)
#          self.w4_max = self.verify_max(self.w4_list)
#          self.vel_max = self.verify_max(self.vel_list)

    def update_fitness(self):
        start_time = time()
#          print("Avaliando a aptidão dos individuos...")
        USE_MULTIHREAD = True

        if USE_MULTIHREAD:
            with Pool() as p:
                val = p.imap(calc_fitness, self.population)
                for result in val:
                    self.update_population(result)
        else:
            for ind in self.population:
                self.update_population(calc_fitness(ind))

        total_time = round(time() - start_time, 3)
        ind_per_s = round(self.N / total_time)
        eta = (self.MAX_I - self.current_it) * self.N / ind_per_s
        print("Levou", total_time, "segundos ---",
              round(ind_per_s), "ind/s --- ETA =",
              round(eta), "segundos =", floor(eta/60), "minutos =",
              floor(eta/3600), "horas", end='\r')

    def run(self):
        best_ind = self.population[0]
        for self.current_it in range(self.MAX_I):
            # print("\nIteration: ", self.current_it)
            # print("\n")
            self.update_fitness()

            prev_best = best_ind
            best_ind = self.best_solution()
            if self.get_fitness(best_ind) != self.get_fitness(prev_best):
                alpha = best_ind.get_alpha()
                beta = best_ind.get_beta()
                print("\nMelhor Solucao:\n Alpha: {}⁰ = {} rad".format(degrees(alpha), alpha),
                      "\nBeta: {}⁰ = {} rad".format(degrees(beta), beta),
                      "\nFitness: ", self.get_fitness(best_ind))

            self.crossover()
            self.mutate()
            self.selection(best_ind)
            self.reset_fitness()

        print("\n**************** RESULTADO ****************")
        alpha = best_ind.get_alpha()
        beta = best_ind.get_beta()

        result = calc_fitness(best_ind)
        best_ind = self.population[result[6]]
        best_ind.fitness_calculated = False
        self.update_population(result)


        print("Alpha: {}⁰ = {} rad".format(degrees(alpha), alpha),
              "\nBeta: {}⁰ = {} rad".format(degrees(beta), beta))
        print("Fitness:\n", self.get_fitness(best_ind))

        print(f"Max value fitness: {self.fitness_max:.2e}")
        print(f"Max value time: {self.time_max:.2e}")
        print(f"Max value w1: {self.w1_max:.2e}")
        print(f"Max value w2: {self.w2_max:.2e}")
        print(f"Max value w3: {self.w3_max:.2e}")
        print(f"Max value w4: {self.w4_max:.2e}")
        print(f"Max value velocity: {self.vel_max:.2e}")

        plt.figure()
#          length_list = len(self.fitness_list)
        # W1 W2
        # W3 W4
        # V  Fit
        plt.subplot(3, 2, 1)
        plt.plot(self.w1_list)
        plt.title("$\\omega_1$")
        plt.grid(True)
        plt.subplot(3, 2, 2)
        plt.plot(self.w2_list)
        plt.title("$\\omega_2$")
        plt.grid(True)
        plt.subplot(3, 2, 3)
        plt.plot(self.w3_list)
        plt.title("$\\omega_3$")
        plt.grid(True)
        plt.subplot(3, 2, 4)
        plt.plot(self.w4_list)
        plt.title("$\\omega_4$")
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.plot(self.vel_list)
        plt.title("$V$")
        plt.grid(True)

        plt.subplot(3, 2, 6)
        plt.plot(self.fitness_list)
        plt.title("Fitness")
        plt.grid(True)

#          plt.stackplot(range(0, length_list), self.fitness_list, self.w1_list,
#                        self.w2_list, self.w3_list, self.w4_list, self.vel_list,
#                        labels=['Fitness', 'w1', 'w2', 'w3', 'w4', 'Velocity'],
#                        colors=['r', 'g', 'b', 'c', 'm', 'y'])
        plt.show()


if __name__ == "__main__":
    ga_solv = GA(prob_cross=0.9, prob_mut=0.10, n_indv=50, n_iterations=1000)
    ga_solv.run()

"""
Código para o problema dos missionários e canibais
"""

from enum import Enum
import numpy as np
import os
from time import sleep

MISSIONARIOS_ROW = 0
CANIBAIS_ROW = 1


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


class POSITION(Enum):
    """
    Posições possíveis
    """
    LEFT_SHORE = 0
    BOAT = 1
    RIGHT_SHORE = 2


# ========================= AÇÕES =============================================
MOVE_MISSIONARIO_FROM_LEFT_TO_BOAT = {'id': 0,
                                      'action': np.asarray([[-1, +1, 0],
                                                            [0, 0, 0]]),
                                      'str': 'Movendo missionário da esquerda para o barco'}
MOVE_MISSIONARIO_FROM_RIGHT_TO_BOAT = {'id': 1,
                                       'action': np.asarray([[0, +1, -1],
                                                             [0, 0, 0]]),
                                       'str': 'Movendo missionário da direita para o barco'}
MOVE_MISSIONARIO_TO_LEFT_SHORE = {'id': 2, 'action': np.asarray([[+1, -1, 0],
                                                                 [0, 0, 0]]),
                                  'str': 'Movendo missionário para a esquerda'}
MOVE_MISSIONARIO_TO_RIGHT_SHORE = {'id': 3, 'action': np.asarray([[0, -1, +1],
                                                                  [0, 0, 0]]),
                                   'str': 'Movendo missionário para a direita'}
MOVE_CANIBAL_FROM_LEFT_TO_BOAT = {'id': 4, 'action': np.asarray([[0, 0, 0],
                                                                 [-1, +1, 0]]),
                                  'str': 'Movendo canibal da esquerda para o barco'}
MOVE_CANIBAL_FROM_RIGHT_TO_BOAT = {'id': 5,
                                   'action': np.asarray([[0, 0, 0],
                                                         [0, +1, -1]]),
                                   'str': 'Movendo canibal da direita para o barco'}
MOVE_CANIBAL_TO_LEFT_SHORE = {'id': 6, 'action': np.asarray([[0, 0, 0],
                                                             [+1, -1, 0]]),
                              'str': 'Movendo canibal para a esquerda'}
MOVE_CANIBAL_TO_RIGHT_SHORE = {'id': 7, 'action': np.asarray([[0, 0, 0],
                                                              [0, -1, +1]]),
                               'str': 'Movendo canibal para a direita'}
MOVE_BOAT = {'id': 8, 'action':  0, 'str': 'Movendo o barco'}
# =============================================================================


class State():
    """
    Estado do problema
    """

    def __init__(self, new_state=None, parent=None):
        if new_state is not None:
            self.state = new_state
        else:
            self.state = np.asarray([[0, 0, 3],  # Missionários
                                     [0, 0, 3]])  # Canibais
        self.boat_on_left = False
        self.parent = parent

    def get_missionarios_on(self, side: POSITION):
        """
        Retorna a quantidade de missionários em um lugar do jogo:
        - Lado esquerdo
        - Barco
        - Lado direito
        """
        return self.state[MISSIONARIOS_ROW][side.value]

    def get_canibais_on(self, side: POSITION):
        """
        Retorna a quantidade de canibais em um lugar do jogo:
        - Lado esquerdo
        - Barco
        - Lado direito
        """
        return self.state[CANIBAIS_ROW][side.value]

    def show_state(self):
        """
        Mostra o estado atual
        """
        aux = ""
        aux += ' ' * 6 + \
               ' ' * (int(not self.boat_on_left) * 2) + \
               'M' * self.get_missionarios_on(POSITION.BOAT) + \
               'C' * self.get_canibais_on(POSITION.BOAT) + \
               ' ' * (2 - self.get_missionarios_on(POSITION.BOAT) -
                      self.get_canibais_on(POSITION.BOAT)) + \
               ' ' * (int(self.boat_on_left) * 2) + \
               ' ' * 6
        print(aux)  # Linha do barco

        boat = '+'
        if self.boat_on_left:
            boat = '-'
        aux = ""
        aux += 'M' * self.get_missionarios_on(POSITION.LEFT_SHORE) + \
               ' ' * (3 - self.get_missionarios_on(POSITION.LEFT_SHORE)) + \
               'C' * self.get_canibais_on(POSITION.LEFT_SHORE) + \
               ' ' * (3 - self.get_canibais_on(POSITION.LEFT_SHORE)) + \
               ' ' * (int(not self.boat_on_left) * 2) + \
               boat * 2 + \
               ' ' * (int(self.boat_on_left) * 2) + \
               'M' * self.get_missionarios_on(POSITION.RIGHT_SHORE) + \
               ' ' * (3 - self.get_missionarios_on(POSITION.RIGHT_SHORE)) + \
               'C' * self.get_canibais_on(POSITION.RIGHT_SHORE) + \
               ' ' * (3 - self.get_canibais_on(POSITION.RIGHT_SHORE))
        print(aux)  # Linha dos missionários e canibais

        aux = ""
        aux += '-' * 6 + ' ' * 4 + '+' * 6
        print(aux)  # Linha dos lados

    def check(self, action):
        """
        Checa se o estado gerado pela ação é válido

        1 - Não podem existir valores negativos na matriz de estados
        2 - Não podem existir valores maiores que 3 na matriz de estados

        3 - Não é possível mover o barco sem missionários ou canibais

        4 - Não é possível existir mais de 2 pessoas no barco

        5 - Não podem existir mais canibais do que missionários do lado
            esquerdo
        6 - Não podem existir mais canibais do que missionários do lado direito

        7 - Não é possível mover o missionário ou canibal para o lado esquerdo
            se o barco não está lá
        8 - Não é possível mover o missionário ou canibal para o lado direito
            se o barco não está lá

        9 - Não é possível mover o missionário ou canibal do lado esquerdo para
            o barco se ele não está lá
        10 - Não é possível mover o missionário ou canibal do lado direito para
            o barco se ele não está lá
        """
        # 1
        if np.min(self.state) < 0:
            return False
        # 2
        if np.max(self.state) > 3:
            return False

        # 3
        if action['id'] is MOVE_BOAT['id'] and \
                (self.get_canibais_on(POSITION.BOAT) == 0 and
                 self.get_missionarios_on(POSITION.BOAT) == 0):
            return False

        # 4
        if self.get_missionarios_on(POSITION.BOAT) + \
                self.get_canibais_on(POSITION.BOAT) > 2:
            return False

        # 5
        if self.get_missionarios_on(POSITION.LEFT_SHORE) < \
                self.get_canibais_on(POSITION.LEFT_SHORE) and \
                self.get_missionarios_on(POSITION.LEFT_SHORE) > 0:
            return False
        # 6
        if self.get_missionarios_on(POSITION.RIGHT_SHORE) < \
                self.get_canibais_on(POSITION.RIGHT_SHORE) and \
                self.get_missionarios_on(POSITION.RIGHT_SHORE) > 0:
            return False

        # 7
        if (action['id'] is MOVE_MISSIONARIO_TO_LEFT_SHORE['id']) or \
                (action['id'] is MOVE_CANIBAL_TO_LEFT_SHORE['id']):
            return self.boat_on_left
        # 8
        if (action['id'] is MOVE_MISSIONARIO_TO_RIGHT_SHORE['id']) or \
                (action['id'] is MOVE_CANIBAL_TO_RIGHT_SHORE['id']):
            return not self.boat_on_left
        # 9
        if (action['id'] is MOVE_MISSIONARIO_FROM_LEFT_TO_BOAT['id']) or \
                (action['id'] is MOVE_CANIBAL_FROM_LEFT_TO_BOAT['id']):
            return self.boat_on_left
        # 10
        if (action['id'] is MOVE_MISSIONARIO_FROM_RIGHT_TO_BOAT['id']) or \
                (action['id'] is MOVE_CANIBAL_FROM_RIGHT_TO_BOAT['id']):
            return not self.boat_on_left
        return True

    def do_action(self, action):
        """
        Realiza uma ação e retorna o novo estado caso ele seja válido
        """
        if action['id'] is MOVE_BOAT['id']:
            new_state = State(self.state, self)
            new_state.boat_on_left = not self.boat_on_left
        else:
            new_state = State(self.state + action['action'], self)
            new_state.boat_on_left = self.boat_on_left

        if new_state.check(action) is True:
            return new_state
        return None

    def is_solution(self):
        """
        Checa se o estado atual é solução
        """
        if self.get_missionarios_on(POSITION.LEFT_SHORE) == 3 and \
                self.get_canibais_on(POSITION.LEFT_SHORE) == 3:
            return True
        return False

    def to_hash(self):
        """
        Converte o estado em um hash para evitar explorar estados já visitados
        """
        str_list = str(np.reshape(self.state, (1, 6))[0].tolist())
        str_list += '{}'.format(self.boat_on_left)
        return hash(str_list,)
# =============================================================================


class BFS():
    """
    Busca em largura
    """

    def __init__(self):
        self.states_list = []
        self.visited_states = []
        self.actions = [MOVE_MISSIONARIO_FROM_LEFT_TO_BOAT,
                        MOVE_MISSIONARIO_FROM_RIGHT_TO_BOAT,
                        MOVE_MISSIONARIO_TO_LEFT_SHORE,
                        MOVE_MISSIONARIO_TO_RIGHT_SHORE,
                        MOVE_CANIBAL_FROM_LEFT_TO_BOAT,
                        MOVE_CANIBAL_FROM_RIGHT_TO_BOAT,
                        MOVE_CANIBAL_TO_LEFT_SHORE,
                        MOVE_CANIBAL_TO_RIGHT_SHORE, MOVE_BOAT]

    def find_solution(self, initial_state: State) -> State:
        """
        Encontra a solução do problema
        """
        self.states_list.append(initial_state)
        solution_found = False
        solution = None
        while len(self.states_list) > 0 and not solution_found:
            current_state = self.states_list.pop(0)
            self.visited_states.append(current_state.to_hash())

            if current_state.is_solution():
                solution_found = True
                solution = current_state
            else:
                for action in self.actions:
                    new_state = current_state.do_action(action)
                    if new_state is not None:
                        if not self.visited_states.__contains__(new_state.to_hash()):
                            self.states_list.append(new_state)
        return solution
# =============================================================================


if __name__ == "__main__":
    bfs = BFS()
    solution = bfs.find_solution(State())
    if solution is not None:
        print("Solução encontrada! Pressione Enter para mostrar!")
        input()
        solution.show_state()
        currente_state = solution
        full_solution = []

        while currente_state is not None:
            full_solution.append(currente_state)
            currente_state = currente_state.parent
        for solution in full_solution[-1::-1]:
            cls()
            solution.show_state()
            sleep(0.5)
        print("Fim")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar seus dados (Simulando o CSV)
# Colunas: [x (lon), y (lat), db (leitura)]
sensores = np.array([
    [10, 10, 80],  # Sensor A
    [50, 10, 75],  # Sensor B
    [30, 40, 85]   # Sensor C (mais perto do estrondo)
])

# Função para converter dB em Energia Linear (Crucial!)
def db_to_linear(db):
    return 10 ** (db / 10)

energia_sensores = db_to_linear(sensores[:, 2])

# 2. Criar a Grade (Grid) do Mapa
# Digamos que nosso mapa vai de X=0 a 60, e Y=0 a 60
resolucao = 1.0 # 1 metro por pixel
x_grid = np.arange(0, 60, resolucao)
y_grid = np.arange(0, 60, resolucao)
X, Y = np.meshgrid(x_grid, y_grid)

# Matriz para guardar o nível de "probabilidade" de cada pixel ser a fonte
probabilidade_mapa = np.zeros(X.shape)

# 3. O Modelo de Localização (Grid Search)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        px, py = X[i, j], Y[i, j]
        
        # Calcular a distância desse pixel para todos os sensores
        distancias = np.sqrt((sensores[:, 0] - px)**2 + (sensores[:, 1] - py)**2)
        
        # Evitar divisão por zero se o pixel cair exatamente em cima do sensor
        distancias[distancias < 1] = 1 
        
        # Estimar qual seria a energia DA FONTE segundo cada sensor
        # Energia da Fonte = Energia Lida * (Distância ^ 2)
        energia_estimada_fonte = energia_sensores * (distancias ** 2)
        
        # Se todos os sensores estimarem a MESMA energia para a fonte, a variância é baixa.
        # Logo, é muito provável que a fonte esteja ali.
        # Usamos o inverso do desvio padrão como nossa "pontuação" de probabilidade (heatmap)
        desvio = np.std(energia_estimada_fonte)
        probabilidade_mapa[i, j] = 1.0 / (desvio + 1e-6) # 1e-6 para evitar divisão por 0

# 4. Renderizar o Mapa de Calor
plt.figure(figsize=(10, 8))
# Usar um colormap escuro estilo "Smart City" (ex: inferno, magma, viridis)
plt.pcolormesh(X, Y, probabilidade_mapa, cmap='inferno', shading='auto')

# Plotar os sensores por cima para referência
plt.scatter(sensores[:, 0], sensores[:, 1], c='cyan', s=100, label='Sensores (Medições)', edgecolors='white')
for s in sensores:
    plt.text(s[0]+1, s[1]+1, f"{s[2]}dB", color='cyan', fontsize=12, fontweight='bold')

plt.colorbar(label='Probabilidade de Localização da Fonte')
plt.title('Mapa de Calor - Localização de Fonte Acústica')
plt.legend()
plt.show()

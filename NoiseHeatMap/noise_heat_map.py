import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import contextily as cx  # <- Nova biblioteca para o mapa real

# 1. Dados dos sensores
sensores = np.array([
    [-46.539173, -23.647839, 45], # Rua das Paineiras x Rua das Goiabeiras
    [-46.541364, -23.646585, 85], # Rua das Figueiras x Alameda São Caetano
    [-46.539614, -23.649242, 60], # Rua das Figueiras x Rua das Pitangueiras
    [-46.539691, -23.646995, 95], # Rua Paineiras x Rua Caneleiras
    [-46.540814, -23.647490, 50]  # Rua das Figueiras x Rua das Caneleiras
])

x_sens = sensores[:, 0]
y_sens = sensores[:, 1]
db_sens = sensores[:, 2]

# 2. Criar a malha com uma margem maior para ver as ruas ao redor
margem = 0.002 # Aumentei um pouco a margem para dar contexto ao bairro
min_x, max_x = x_sens.min() - margem, x_sens.max() + margem
min_y, max_y = y_sens.min() - margem, y_sens.max() + margem

grid_x, grid_y = np.mgrid[min_x:max_x:300j, min_y:max_y:300j]

# 3. Interpolação RBF
epsilon_geografico = 0.001 
rbf_interpolator = Rbf(x_sens, y_sens, db_sens, function='inverse', epsilon=epsilon_geografico)

# 4. Calcular dBs
db_grid = rbf_interpolator(grid_x, grid_y)
db_grid = np.clip(db_grid, a_min=20, a_max=120)

# 5. Renderização
fig, ax = plt.subplots(figsize=(12, 10))

# Remover o fundo preto manual, pois o mapa será o fundo
ax.ticklabel_format(useOffset=False, style='plain')

# Renderizar o gradiente. O alpha=0.55 é o segredo da transparência
mesh = ax.pcolormesh(grid_x, grid_y, db_grid, cmap='turbo', shading='auto', alpha=0.35, zorder=2)

# Plotar os sensores
ax.scatter(x_sens, y_sens, c='white', s=80, edgecolors='black', zorder=5)

for i in range(len(sensores)):
    ax.text(x_sens[i] + 0.0001, y_sens[i] + 0.0001, f"{int(db_sens[i])} dB", 
             color='white', fontweight='bold', fontsize=11, 
             path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black')],
             zorder=6)

# --- A MÁGICA ACONTECE AQUI ---
# Adiciona o mapa real por baixo das coordenadas geográficas
# O provider 'CartoDB.DarkMatter' dá o exato tom de dashboard do seu mockup
cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.DarkMatter, zorder=1)
# ------------------------------

cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Nível de Ruído (dB)', color='black', fontsize=12, fontweight='bold')

plt.title('Dashboard de Poluição Sonora - Bairro Jardim', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

plt.show()

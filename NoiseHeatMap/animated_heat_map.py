# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from scipy.interpolate import Rbf
# import contextily as cx
# import pytz # Biblioteca para lidar com fusos horários
# import xml.etree.ElementTree as ET # Para ler o arquivo GPX
#
# # filename = '~/SocialDroids/NoiseSensor_docs/MetricasBairroJardim/noise_data_2026-06-15T11_34_00.000Z_2026-06-20T11_34_00.000Z.csv'
# # noise_filename = '~/SocialDroids/NoiseSensor_docs/MetricasBairroJardim/noise_data_2026-06-19T09_00_00.000Z_2026-06-20T09_00_00.000Z-dia_jogo_brasil.csv'
# noise_filename = '~/SocialDroids/NoiseSensor_docs/MetricasBairroJardim/noise_data_jogo_brasil_buzinaco.csv'
# gps_filename = '../../SocialDroids/NoiseSensor_docs/MetricasBairroJardim/Santo_Andre_noise_sensor.gpx'
#
# # 1. Definição dos Sensores (JSON original)
# sensores_json = {
#     "26aab129e5cf4849": {"latitude": -23.647839, "longitude": -46.539173, "label": "Rua das Paineiras x Rua das Goiabeiras"},
#     "f356cd540573e894": {"latitude": -23.649242, "longitude": -46.539614, "label": "Rua das Figueiras x Rua das Pitangueiras"},
#     "21954070b67064bd": {"latitude": -23.646995, "longitude": -46.539691, "label": "Rua Paineiras x Rua Caneleiras"},
#     "8c9da241fc5f95de": {"latitude": -23.647490, "longitude": -46.540814, "label": "Rua das Figueiras x Rua das Caneleiras"},
#     "ab5918f52d85a691": {"latitude": -23.645597, "longitude": -46.541940, "label": "Rua das Figueiras x Alameda São Caetano"}
# , 
# }
#
# # Definindo o fuso horário de destino (Brasília - GMT-3)
# tz_br = pytz.timezone('America/Sao_Paulo')
#
# # 2. Função Acústica: Cálculo do Leq (Nível Contínuo Equivalente)
# def calcular_leq(db_series):
#     if db_series.empty or db_series.isna().all():
#         return np.nan
#     # Converte os dB do período para energia linear, tira a média, e volta para dB
#     energia = 10 ** (db_series / 10)
#     media_energia = np.mean(energia)
#     return 10 * np.log10(media_energia)
#
# # 3. Carregar e Processar os Dados (Resampling do CSV)
# print("Processando o arquivo CSV de ruído...")
# df = pd.read_csv(noise_filename)
# df['time'] = pd.to_datetime(df['time'], utc=True, format='ISO8601')
# df.set_index('time', inplace=True)
#
# janela_tempo = '1S'
# df_resampled = df.groupby('dev_eui')['noise_data'].resample(janela_tempo).apply(calcular_leq).unstack(level=0)
# df_resampled = df_resampled.bfill().ffill()
#
# # 4. Carregar e Processar o GPX (Trajeto WikiLoc)
# print("Processando o arquivo GPX do trajeto...")
# # O namespace padrão dos arquivos GPX do Wikiloc
# ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
# tree = ET.parse(gps_filename) # <-- CAMINHO DO SEU ARQUIVO GPX
# root = tree.getroot()
#
# gpx_data = []
# for trkpt in root.findall('.//gpx:trkpt', ns):
#     lat = float(trkpt.attrib['lat'])
#     lon = float(trkpt.attrib['lon'])
#     time_str = trkpt.find('gpx:time', ns).text
#     gpx_data.append({'time': time_str, 'lat': lat, 'lon': lon})
#
# df_gpx = pd.DataFrame(gpx_data)
# # Converte as strings de tempo do GPX garantindo que são UTC (+0)
# df_gpx['time'] = pd.to_datetime(df_gpx['time'], utc=True)
# df_gpx.set_index('time', inplace=True)
#
# # MÁGICA DA SINCRONIZAÇÃO: Interpola o caminho GPX para a mesma linha do tempo dos sensores
# combined_index = df_gpx.index.union(df_resampled.index).sort_values()
# df_gpx_interp = df_gpx.reindex(combined_index).interpolate(method='time')
# # Extrai apenas as coordenadas exatas correspondentes aos frames da nossa animação
# df_gpx_frames = df_gpx_interp.reindex(df_resampled.index)
#
# # 5. Preparar Coordenadas da Malha (Grid)
# euis_ativos = [eui for eui in df_resampled.columns if eui in sensores_json]
# x_sens = np.array([sensores_json[eui]['longitude'] for eui in euis_ativos])
# y_sens = np.array([sensores_json[eui]['latitude'] for eui in euis_ativos])
#
# # Garantir que o mapa abranja tanto os sensores quanto o trajeto GPX
# margem = 0.002
# min_x = min(x_sens.min(), df_gpx['lon'].min()) - margem
# max_x = max(x_sens.max(), df_gpx['lon'].max()) + margem
# min_y = min(y_sens.min(), df_gpx['lat'].min()) - margem
# max_y = max(y_sens.max(), df_gpx['lat'].max()) + margem
#
# # Redução da resolução para 150j para gerar o vídeo muito mais rápido nos testes
# grid_x, grid_y = np.mgrid[min_x:max_x:150j, min_y:max_y:150j]
#
# # 6. Configurar a Figura e o Mapa Base (TEMA CLARO)
# print("Baixando mapa base geográfico (OpenStreetMap)...")
# fig, ax = plt.subplots(figsize=(12, 10))
#
# # Alterando tudo para fundo branco e textos pretos (Tema Claro)
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')
# ax.ticklabel_format(useOffset=False, style='plain')
#
# ax.set_xlim(min_x, max_x)
# ax.set_ylim(min_y, max_y)
#
# # Usando o visual padrão claro (OpenStreetMap)
# cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.Mapnik, zorder=1, zoom=16)
#
# # --- OTIMIZAÇÃO EXTREMA: Pré-computação das matrizes RBF ---
# print("A pré-computar matrizes de interpolação (Aceleração RBF)...")
# epsilon_rbf = 0.0005
# num_sens = len(x_sens)
#
# # 1. Matriz de distâncias entre os próprios sensores (Matriz A)
# A = np.zeros((num_sens, num_sens))
# for i in range(num_sens):
#     for j in range(num_sens):
#         r = np.sqrt((x_sens[i] - x_sens[j])**2 + (y_sens[i] - y_sens[j])**2)
#         A[i, j] = 1.0 / np.sqrt((r / epsilon_rbf)**2 + 1.0)
#
# A_inv = np.linalg.inv(A) # Inversa da matriz A calculada apenas 1x
#
# # 2. Matriz de distâncias dos sensores para TODOS os pontos da grelha
# grid_flat_x = grid_x.ravel()
# grid_flat_y = grid_y.ravel()
# Phi_grid = np.zeros((num_sens, len(grid_flat_x)))
#
# for i in range(num_sens):
#     r = np.sqrt((grid_flat_x - x_sens[i])**2 + (grid_flat_y - y_sens[i])**2)
#     Phi_grid[i, :] = 1.0 / np.sqrt((r / epsilon_rbf)**2 + 1.0)
# # -----------------------------------------------------------
#
# # 7. Elementos que serão animados
# db_sens_inicial = df_resampled[euis_ativos].iloc[0].values
#
# # Interpolação vetorizada para o primeiro frame (Substitui o Rbf)
# w_inicial = A_inv.dot(db_sens_inicial)
# db_grid_init = w_inicial.dot(Phi_grid).reshape(grid_x.shape)
#
# # Mantemos o alpha do mapa de calor em 0.35 para não esconder as ruas do OSM
# mesh_plot = ax.pcolormesh(grid_x, grid_y, db_grid_init,
#                           cmap='turbo', shading='auto', alpha=0.35, zorder=2, vmin=40, vmax=100)
#
# # Trilha estática do GPX (Linha azul pontilhada de fundo)
# ax.plot(df_gpx['lon'], df_gpx['lat'], color='blue', linewidth=2, linestyle='--', alpha=0.6, zorder=3, label='Trajeto Percorrido')
#
# # Marcador animado do GPS (Sua posição atual)
# gps_marker = ax.scatter([], [], c='red', s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=7, label='Emissor (Você)')
#
# # Pontos dos sensores
# ax.scatter(x_sens, y_sens, c='white', s=80, edgecolors='black', zorder=5, label='Sensores')
#
# # Legenda para explicar o que é a estrela e a linha
# ax.legend(loc='upper right', framealpha=0.9)
#
# # Textos com os rótulos de dB
# textos_sensores = []
# for i in range(len(euis_ativos)):
#     txt = ax.text(x_sens[i] + 0.0001, y_sens[i] + 0.0001, "",
#                   color='white', fontweight='bold', fontsize=11,
#                   path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black')],
#                   zorder=6)
#     textos_sensores.append(txt)
#
# # Configuração da Barra de Cores (Textos escuros agora)
# cbar = plt.colorbar(mesh_plot, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label('Nível de Ruído (Leq dB)', color='black', fontsize=12, fontweight='bold')
# cbar.ax.yaxis.set_tick_params(color='black')
# plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
#
# titulo = ax.set_title('', fontsize=14, fontweight='bold', pad=20, color='black')
# ax.set_xlabel('Longitude', color='black')
# ax.set_ylabel('Latitude', color='black')
# ax.tick_params(colors='black')
#
# # 8. Função de Atualização do Frame
# def atualizar_frame(frame_idx):
#     row = df_resampled[euis_ativos].iloc[frame_idx]
#     db_sens = row.values
#     tempo_utc = df_resampled.index[frame_idx]
#
#     # Converte o tempo de UTC para o fuso GMT-3
#     tempo_br = tempo_utc.astimezone(tz_br)
#
#     # --- OTIMIZAÇÃO APLICADA ---
#     # Multiplicação simples de matrizes em vez de recriar a Rbf
#     w = A_inv.dot(db_sens)
#     db_grid = w.dot(Phi_grid).reshape(grid_x.shape)
#     
#     # Mantém a escala de cores estável
#     db_grid = np.clip(db_grid, a_min=40, a_max=100)
#     mesh_plot.set_array(db_grid.ravel())
#
#     # Atualiza textos
#     for i, txt in enumerate(textos_sensores):
#         txt.set_text(f"{int(db_sens[i])} dB")
#
#     # ATUALIZAÇÃO DO GPS
#     # Pega as coordenadas interpoladas correspondentes a esse exato bloco de 30s
#     current_lat = df_gpx_frames.iloc[frame_idx]['lat']
#     current_lon = df_gpx_frames.iloc[frame_idx]['lon']
#     
#     # Se naquele momento o GPS estava gravando (ou seja, os dados interpolados não são nulos)
#     if not pd.isna(current_lat) and not pd.isna(current_lon):
#         gps_marker.set_visible(True)
#         gps_marker.set_offsets(np.c_[current_lon, current_lat])
#     else:
#         gps_marker.set_visible(False)
#
#     titulo.set_text(f"Dinâmica de Poluição Sonora vs Emissor\nTempo: {tempo_br.strftime('%H:%M:%S')} (GMT-3)")
#
#     return [mesh_plot, titulo, gps_marker] + textos_sensores
#
# # 9. Cria e Roda a Animação
# print(f"Gerando animação para {len(df_resampled)} frames (bloquinhos de {janela_tempo})...")
# animacao = animation.FuncAnimation(fig, atualizar_frame,
#                                    frames=len(df_resampled),
#                                    interval=500,
#                                    blit=True)
#
# print("Salvando animação em MP4...")
# FFwriter = animation.FFMpegWriter(fps=10) # 10 FPS costuma ser o ideal para vídeos de séries temporais
# animacao.save(f'mapa_ruido_gmt3_com_gps_{janela_tempo}_cut.mp4', writer=FFwriter)
# print("Concluído!")
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import Rbf
import contextily as cx
import pytz # Biblioteca para lidar com fusos horários
import xml.etree.ElementTree as ET # Para ler o arquivo GPX

# filename = '~/SocialDroids/NoiseSensor_docs/MetricasBairroJardim/noise_data_2026-06-15T11_34_00.000Z_2026-06-20T11_34_00.000Z.csv'
# noise_filename = '~/SocialDroids/NoiseSensor_docs/MetricasBairroJardim/noise_data_2026-06-19T09_00_00.000Z_2026-06-20T09_00_00.000Z-dia_jogo_brasil.csv'
noise_filename = '~/SocialDroids/NoiseSensor_docs/MetricasBairroJardim/noise_data_jogo_brasil_buzinaco.csv'
gps_filename = '../../SocialDroids/NoiseSensor_docs/MetricasBairroJardim/Santo_Andre_noise_sensor.gpx'

# 1. Definição dos Sensores (JSON original)
sensores_json = {
    "26aab129e5cf4849": {"latitude": -23.647839, "longitude": -46.539173, "label": "Rua das Paineiras x Rua das Goiabeiras"},
    "f356cd540573e894": {"latitude": -23.649242, "longitude": -46.539614, "label": "Rua das Figueiras x Rua das Pitangueiras"},
    "21954070b67064bd": {"latitude": -23.646995, "longitude": -46.539691, "label": "Rua Paineiras x Rua Caneleiras"},
    "8c9da241fc5f95de": {"latitude": -23.647490, "longitude": -46.540814, "label": "Rua das Figueiras x Rua das Caneleiras"},
    "ab5918f52d85a691": {"latitude": -23.645597, "longitude": -46.541940, "label": "Rua das Figueiras x Alameda São Caetano"}
, 
}

# Definindo o fuso horário de destino (Brasília - GMT-3)
tz_br = pytz.timezone('America/Sao_Paulo')

# 2. Função Acústica: Cálculo do Leq (Nível Contínuo Equivalente)
def calcular_leq(db_series):
    if db_series.empty or db_series.isna().all():
        return np.nan
    # Converte os dB do período para energia linear, tira a média, e volta para dB
    energia = 10 ** (db_series / 10)
    media_energia = np.mean(energia)
    return 10 * np.log10(media_energia)

# 3. Carregar e Processar os Dados (Resampling do CSV)
print("Processando o arquivo CSV de ruído...")
df = pd.read_csv(noise_filename)
df['time'] = pd.to_datetime(df['time'], utc=True, format='ISO8601')
df.set_index('time', inplace=True)

janela_tempo = '1S'
df_resampled = df.groupby('dev_eui')['noise_data'].resample(janela_tempo).apply(calcular_leq).unstack(level=0)
df_resampled = df_resampled.bfill().ffill()

# 4. Carregar e Processar o GPX (Trajeto WikiLoc)
print("Processando o arquivo GPX do trajeto...")
# O namespace padrão dos arquivos GPX do Wikiloc
ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
tree = ET.parse(gps_filename) # <-- CAMINHO DO SEU ARQUIVO GPX
root = tree.getroot()

gpx_data = []
for trkpt in root.findall('.//gpx:trkpt', ns):
    lat = float(trkpt.attrib['lat'])
    lon = float(trkpt.attrib['lon'])
    time_str = trkpt.find('gpx:time', ns).text
    gpx_data.append({'time': time_str, 'lat': lat, 'lon': lon})

df_gpx = pd.DataFrame(gpx_data)
# Converte as strings de tempo do GPX garantindo que são UTC (+0)
df_gpx['time'] = pd.to_datetime(df_gpx['time'], utc=True)
df_gpx.set_index('time', inplace=True)

# --- NOVO: PARÂMETRO DE SINCRONIZAÇÃO DO GPS ---
# Se o GPS parece "adiantado" (chega ao local antes de o gráfico de ruído reagir),
# use um valor positivo para o atrasar (ex: 15). Se estiver atrasado, use um valor negativo.
atraso_gps_segundos = 20 
if atraso_gps_segundos != 0:
    print(f"A ajustar o tempo do GPS em {atraso_gps_segundos} segundos...")
    df_gpx.index = df_gpx.index + pd.Timedelta(seconds=atraso_gps_segundos)
# -----------------------------------------------

# MÁGICA DA SINCRONIZAÇÃO: Interpola o caminho GPX para a mesma linha do tempo dos sensores
combined_index = df_gpx.index.union(df_resampled.index).sort_values()
df_gpx_interp = df_gpx.reindex(combined_index).interpolate(method='time')
# Extrai apenas as coordenadas exatas correspondentes aos frames da nossa animação
df_gpx_frames = df_gpx_interp.reindex(df_resampled.index)

# 5. Preparar Coordenadas da Malha (Grid)
euis_ativos = [eui for eui in df_resampled.columns if eui in sensores_json]
x_sens = np.array([sensores_json[eui]['longitude'] for eui in euis_ativos])
y_sens = np.array([sensores_json[eui]['latitude'] for eui in euis_ativos])

# Garantir que o mapa abranja tanto os sensores quanto o trajeto GPX
margem = 0.002
min_x = min(x_sens.min(), df_gpx['lon'].min()) - margem
max_x = max(x_sens.max(), df_gpx['lon'].max()) + margem
min_y = min(y_sens.min(), df_gpx['lat'].min()) - margem
max_y = max(y_sens.max(), df_gpx['lat'].max()) + margem

# Redução da resolução para 150j para gerar o vídeo muito mais rápido nos testes
grid_x, grid_y = np.mgrid[min_x:max_x:150j, min_y:max_y:150j]

# 6. Configurar a Figura e o Mapa Base (TEMA CLARO)
print("Baixando mapa base geográfico (OpenStreetMap)...")
fig, ax = plt.subplots(figsize=(12, 10))

# Alterando tudo para fundo branco e textos pretos (Tema Claro)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.ticklabel_format(useOffset=False, style='plain')

ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

# Usando o visual padrão claro (OpenStreetMap)
cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.Mapnik, zorder=1, zoom=16)

# --- OTIMIZAÇÃO EXTREMA: Pré-computação das matrizes RBF ---
print("A pré-computar matrizes de interpolação (Aceleração RBF)...")
epsilon_rbf = 0.0005
num_sens = len(x_sens)

# 1. Matriz de distâncias entre os próprios sensores (Matriz A)
A = np.zeros((num_sens, num_sens))
for i in range(num_sens):
    for j in range(num_sens):
        r = np.sqrt((x_sens[i] - x_sens[j])**2 + (y_sens[i] - y_sens[j])**2)
        A[i, j] = 1.0 / np.sqrt((r / epsilon_rbf)**2 + 1.0)

A_inv = np.linalg.inv(A) # Inversa da matriz A calculada apenas 1x

# 2. Matriz de distâncias dos sensores para TODOS os pontos da grelha
grid_flat_x = grid_x.ravel()
grid_flat_y = grid_y.ravel()
Phi_grid = np.zeros((num_sens, len(grid_flat_x)))

for i in range(num_sens):
    r = np.sqrt((grid_flat_x - x_sens[i])**2 + (grid_flat_y - y_sens[i])**2)
    Phi_grid[i, :] = 1.0 / np.sqrt((r / epsilon_rbf)**2 + 1.0)
# -----------------------------------------------------------

# 7. Elementos que serão animados
db_sens_inicial = df_resampled[euis_ativos].iloc[0].values

# Interpolação vetorizada para o primeiro frame (Substitui o Rbf)
w_inicial = A_inv.dot(db_sens_inicial)
db_grid_init = w_inicial.dot(Phi_grid).reshape(grid_x.shape)

# Mantemos o alpha do mapa de calor em 0.35 para não esconder as ruas do OSM
mesh_plot = ax.pcolormesh(grid_x, grid_y, db_grid_init,
                          cmap='turbo', shading='auto', alpha=0.35, zorder=2, vmin=40, vmax=100)

# Trilha estática do GPX (Linha azul pontilhada de fundo)
ax.plot(df_gpx['lon'], df_gpx['lat'], color='blue', linewidth=2, linestyle='--', alpha=0.6, zorder=3, label='Trajeto Percorrido')

# Marcador animado do GPS (Sua posição atual)
gps_marker = ax.scatter([], [], c='red', s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=7, label='Emissor (Você)')

# Pontos dos sensores
ax.scatter(x_sens, y_sens, c='white', s=80, edgecolors='black', zorder=5, label='Sensores')

# Legenda para explicar o que é a estrela e a linha
ax.legend(loc='upper right', framealpha=0.9)

# Textos com os rótulos de dB
textos_sensores = []
for i in range(len(euis_ativos)):
    txt = ax.text(x_sens[i] + 0.0001, y_sens[i] + 0.0001, "",
                  color='white', fontweight='bold', fontsize=11,
                  path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black')],
                  zorder=6)
    textos_sensores.append(txt)

# Configuração da Barra de Cores (Textos escuros agora)
cbar = plt.colorbar(mesh_plot, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Nível de Ruído (Leq dB)', color='black', fontsize=12, fontweight='bold')
cbar.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

titulo = ax.set_title('', fontsize=14, fontweight='bold', pad=20, color='black')
ax.set_xlabel('Longitude', color='black')
ax.set_ylabel('Latitude', color='black')
ax.tick_params(colors='black')

# 8. Função de Atualização do Frame
def atualizar_frame(frame_idx):
    row = df_resampled[euis_ativos].iloc[frame_idx]
    db_sens = row.values
    tempo_utc = df_resampled.index[frame_idx]

    # Converte o tempo de UTC para o fuso GMT-3
    tempo_br = tempo_utc.astimezone(tz_br)

    # --- OTIMIZAÇÃO APLICADA ---
    # Multiplicação simples de matrizes em vez de recriar a Rbf
    w = A_inv.dot(db_sens)
    db_grid = w.dot(Phi_grid).reshape(grid_x.shape)
    
    # Mantém a escala de cores estável
    db_grid = np.clip(db_grid, a_min=40, a_max=100)
    mesh_plot.set_array(db_grid.ravel())

    # Atualiza textos
    for i, txt in enumerate(textos_sensores):
        txt.set_text(f"{int(db_sens[i])} dB")

    # ATUALIZAÇÃO DO GPS
    # Pega as coordenadas interpoladas correspondentes a esse exato bloco de 30s
    current_lat = df_gpx_frames.iloc[frame_idx]['lat']
    current_lon = df_gpx_frames.iloc[frame_idx]['lon']
    
    # Se naquele momento o GPS estava gravando (ou seja, os dados interpolados não são nulos)
    if not pd.isna(current_lat) and not pd.isna(current_lon):
        gps_marker.set_visible(True)
        gps_marker.set_offsets(np.c_[current_lon, current_lat])
    else:
        gps_marker.set_visible(False)

    titulo.set_text(f"Dinâmica de Poluição Sonora vs Emissor\nTempo: {tempo_br.strftime('%H:%M:%S')} (GMT-3)")

    return [mesh_plot, titulo, gps_marker] + textos_sensores

# 9. Cria e Roda a Animação
print(f"Gerando animação para {len(df_resampled)} frames (bloquinhos de {janela_tempo})...")
animacao = animation.FuncAnimation(fig, atualizar_frame,
                                   frames=len(df_resampled),
                                   interval=500,
                                   blit=True)

print("Salvando animação em MP4...")
FFwriter = animation.FFMpegWriter(fps=2) # 10 FPS costuma ser o ideal para vídeos de séries temporais
animacao.save(f'mapa_ruido_gmt3_com_gps_{janela_tempo}_cut.mp4', writer=FFwriter)
print("Concluído!")

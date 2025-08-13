import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FFMpegWriter
from shapely.geometry import Point, Polygon

############################## Conversão para Unidades SI ######################
# Fatores de conversão
kgf_cm2_to_Pa = 98066.5    # 1 kgf/cm² = 98066.5 Pa
md_to_m2 = 9.869233e-16    # 1 mD = 9.869233e-16 m²
cp_to_Pa_s = 0.001          # 1 cP = 0.001 Pa·s

############################## Reservatório ####################################
Lx = 7000  # metros
Ly = 5000  # metros
ct = 9e-6 / kgf_cm2_to_Pa  # Pa⁻¹ (convertido de (kgf/cm²)⁻¹)
phi = 0.23  # adimensional
mi = 2 * cp_to_Pa_s        # Pa·s (convertido de cP)
P0 = 300 * kgf_cm2_to_Pa   # Pa (convertido de kgf/cm²)
Pwf_prod = 200 * kgf_cm2_to_Pa  # Pa (convertido de kgf/cm²)
k = 500 * md_to_m2         # m² (convertido de mD)
rw = 0.10  # metros
beta = (1 / (mi * phi * ct))  # Fator beta (calculado com unidades SI) em seg
h = 25  # metros (espessura do reservatório)
salvamento = 100  # segundos (tempo de salvamento)
################################################################################

############################ Discretização #####################################
Nx = 1000
Ny = 1000
dt = 1000  # segundos
t_max = 10*3600  # segundos
dx = Lx/Nx
dy = Ly/Ny
rx = dt / (dx**2)  # s/m²
ry = dt / (dy**2)  # s/m²
################################################################################


############################### Matriz auxiliar ################################
A = np.ones([Nx, Ny])
indice = A.copy()
value = 0
for i in range(0, Nx):  # começa da última linha até a primeira
    for j in range(0, Ny):        # esquerda para direita
        indice[i, j] = value
        value += 1

df = pd.DataFrame(indice)
# print("Matriz de índices:")
# print(df)
################################################################################

######################## Matriz das permeabilidades ############################

contorno = [
    (222.40802675585286, 3579.175704989154),
    (397.9933110367893, 3828.633405639913),
    (620.4013377926422, 4164.859002169197),
    (690.6354515050167, 4501.084598698481),
    (877.9264214046823, 4837.310195227766),
    (1030.1003344481605, 4848.156182212581),
    (1076.923076923077, 4685.466377440347),
    (1392.9765886287626, 4783.080260303687),
    (1638.7959866220735, 4718.004338394793),
    (1931.4381270903011, 4609.544468546637),
    (2294.314381270903, 4479.39262472885),
    (2551.839464882943, 4403.47071583514),
    (3078.5953177257525, 4522.776572668113),
    (3230.769230769231, 4349.240780911063),
    (3359.5317725752507, 4262.472885032537),
    (3488.294314381271, 4121.475054229934),
    (3722.408026755853, 4099.783080260303),
    (3816.0535117056857, 3915.401301518438),
    (3933.1103678929767, 3817.787418655097),
    (4050.1672240802677, 3731.0195227765726),
    (4167.224080267559, 3676.789587852494),
    (4307.692307692308, 3687.63557483731),
    (4448.1605351170565, 3752.7114967462035),
    (4576.923076923077, 3774.403470715835),
    (4682.274247491639, 3752.7114967462035),
    (4787.625418060201, 3655.0976138828632),
    (5080.267558528428, 3449.023861171366),
    (5443.1438127090305, 3405.639913232104),
    (5536.789297658863, 3373.101952277657),
    (5595.3177257525085, 3275.4880694143167),
    (5688.963210702341, 3264.642082429501),
    (5735.785953177257, 3177.874186550976),
    (5923.076923076923, 3112.7982646420824),
    (6637.123745819398, 2472.885032537961),
    (6742.47491638796, 2440.3470715835138),
    (6590.301003344482, 2266.811279826464),
    (6438.127090301004, 2223.4273318872015),
    (6356.1872909699, 2125.813449023861),
    (6227.42474916388, 2060.737527114967),
    (6040.133779264214, 1984.815618221258),
    (5899.665551839465, 1854.6637744034706),
    (5747.491638795987, 1724.511930585683),
    (5618.729096989967, 1616.052060737527),
    (5431.438127090301, 1409.9783080260302),
    (5314.38127090301, 1268.9804772234272),
    (5162.207357859532, 1193.058568329718),
    (4939.799331103679, 1247.288503253796),
    (4822.742474916388, 1095.4446854663775),
    (4541.80602006689, 900.2169197396962),
    (4506.688963210702, 639.9132321041214),
    (4471.571906354515, 444.6854663774403),
    (4413.04347826087, 336.22559652928413),
    (4108.695652173913, 553.1453362255965),
    (3640.4682274247493, 932.7548806941431),
    (3347.826086956522, 1182.2125813449022),
    (3043.4782608695655, 1225.5965292841647),
    (2774.247491638796, 1355.7483731019522),
    (2610.367892976589, 1409.9783080260302),
    (2434.782608695652, 1550.9761388286333),
    (2364.5484949832776, 1681.1279826464206),
    (2224.0802675585282, 1778.7418655097613),
    (2083.6120401337794, 1670.2819956616052),
    (1908.0267558528428, 1876.3557483731017),
    (1732.4414715719063, 2147.505422993492),
    (1392.9765886287626, 2960.9544468546637),
    (1311.036789297659, 3091.106290672451),
    (1229.0969899665552, 3188.7201735357917),
    (1018.3946488294315, 3253.7960954446853),
    (772.5752508361204, 3340.5639913232103),
    (526.7558528428094, 3383.9479392624726),
    (222.40802675585286, 3579.175704989154)
]

poly = Polygon(contorno)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)


# Permeabilidade base e interna
k_out = np.nan
k_in = 500*md_to_m2  # m²
k_in_high = 1000 * md_to_m2  # m² (convertido de mD)

# Matriz de permeabilidade
K = np.full((Nx, Ny), k_out)
for i in range(0, len(X)):
    for j in range(0, len(Y)):
        p = Point(X[i, j], Y[i, j])
        if poly.contains(p):
            K[i, j] = k_in
            if X[i, j] > 4000:
                K[i, j] = k_in_high

# K = np.full((Nx, Ny), k_in_high)
permeabilidades = K.copy()
plt.figure(figsize=(6, 5))
plt.imshow(
    permeabilidades.reshape(Nx, Ny),
    cmap='viridis',
    origin='lower',
    extent=[0, Lx, 0, Ly],
)
plt.colorbar(label='Permeabilidade (m²)')
plt.title("Distribuição Da permeabilidade do Reservatório")
title = f'Permeabilidade no Reservatório  dt = {dt} s, Nx = {Nx}, Ny = {Ny}'
# plt.savefig(f'{title}.png', dpi=300)
plt.show()
plt.close()

# ################################################################################


# #################################### Poços #####################################
req = 0.198*dx
indice_poco = int(Nx/2)
k_poco = permeabilidades[indice_poco, indice_poco]
J = (2 * np.pi * k_poco * h) / (mi * np.log(req / rw))
# ###############################################################################


# ########################### Matriz pentaseggonal ###############################

A_ = np.zeros([Ny*Nx, Ny*Nx])
F = np.zeros(Ny*Nx)

i_0 = 0
for j in range(0, Nx):
    for i in range(0, Ny):

        if j == indice_poco and i == indice_poco:
            transmissibilidade_x_leste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i+1]))
            transmissibilidade_x_oeste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i-1]))
            transmissibilidade_y_sul = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j+1, i]))
            transmissibilidade_y_norte = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j-1, i]))

            eta = (permeabilidades[j, i] / (phi*mi*ct))
            epsilon = ((2*np.pi*eta) / (dx*dy*np.log(req/rw))) * dt

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx*transmissibilidade_x_oeste) + \
                (beta*ry*transmissibilidade_y_sul) + \
                (beta*ry*transmissibilidade_y_norte) + epsilon
            A_[i_0, int(indice[j, i+1])] = - \
                (beta*rx*transmissibilidade_x_leste)
            A_[i_0, int(indice[j, i-1])] = - \
                (beta*rx*transmissibilidade_x_oeste)
            A_[i_0, int(indice[j-1, i])] = - \
                (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j+1, i])] = -(beta*ry*transmissibilidade_y_sul)

            F[i_0] = epsilon * Pwf_prod

        elif i == 0 and j == 0:
            transmissibilidade_x_oeste = 0
            transmissibilidade_x_leste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i+1]))
            transmissibilidade_y_norte = 0
            transmissibilidade_y_sul = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j+1, i]))

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx *
                                                                                     transmissibilidade_x_oeste) + (beta*ry*transmissibilidade_y_sul) + (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j, i+1])] = - \
                (beta*rx*transmissibilidade_x_leste)
            A_[i_0, int(indice[j+1, i])] = -(beta*ry*transmissibilidade_y_sul)
            F[i_0] = 0

        elif j == 0 and i == Ny-1:
            transmissibilidade_x_leste = 0
            transmissibilidade_x_oeste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i-1]))
            transmissibilidade_y_sul = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j+1, i]))
            transmissibilidade_y_norte = 0

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx *
                                                                                     transmissibilidade_x_oeste) + (beta*ry*transmissibilidade_y_sul) + (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j, i-1])] = - \
                (beta*rx*transmissibilidade_x_oeste)
            A_[i_0, int(indice[j+1, i])] = -(beta*ry*transmissibilidade_y_sul)
            F[i_0] = 0

        elif j == Nx-1 and i == 0:
            transmissibilidade_x_oeste = 0
            transmissibilidade_x_leste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i+1]))
            transmissibilidade_y_norte = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j-1, i]))
            transmissibilidade_y_sul = 0

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx *
                                                                                     transmissibilidade_x_oeste) + (beta*ry*transmissibilidade_y_sul) + (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j, i+1])] = - \
                (beta*rx*transmissibilidade_x_leste)
            A_[i_0, int(indice[j-1, i])] = - \
                (beta*ry*transmissibilidade_y_norte)
            F[i_0] = 0

        elif j == Nx-1 and i == Ny-1:
            transmissibilidade_x_leste = 0
            transmissibilidade_x_oeste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i-1]))
            transmissibilidade_y_norte = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j-1, i]))
            transmissibilidade_y_sul = 0

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx *
                                                                                     transmissibilidade_x_oeste) + (beta*ry*transmissibilidade_y_sul) + (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j, i-1])] = - \
                (beta*rx*transmissibilidade_x_oeste)
            A_[i_0, int(indice[j-1, i])] = - \
                (beta*ry*transmissibilidade_y_norte)
            F[i_0] = 0

        elif j == 0 and i > 0 and i < Ny-1:
            transmissibilidade_x_leste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i+1]))
            transmissibilidade_x_oeste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i-1]))
            transmissibilidade_y_sul = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j+1, i]))
            transmissibilidade_y_norte = 0

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx *
                                                                                     transmissibilidade_x_oeste) + (beta*ry*transmissibilidade_y_sul) + (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j, i+1])] = - \
                (beta*rx*transmissibilidade_x_leste)
            A_[i_0, int(indice[j, i-1])] = - \
                (beta*rx*transmissibilidade_x_oeste)
            A_[i_0, int(indice[j+1, i])] = -(beta*ry*transmissibilidade_y_sul)
            F[i_0] = 0

        elif i == 0 and j > 0 and j < Nx-1:
            transmissibilidade_x_leste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i+1]))
            transmissibilidade_x_oeste = 0
            transmissibilidade_y_sul = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j+1, i]))
            transmissibilidade_y_norte = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j-1, i]))

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx *
                                                                                     transmissibilidade_x_oeste) + (beta*ry*transmissibilidade_y_sul) + (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j, i+1])] = - \
                (beta*rx*transmissibilidade_x_leste)
            A_[i_0, int(indice[j-1, i])] = - \
                (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j+1, i])] = -(beta*ry*transmissibilidade_y_sul)
            F[i_0] = 0

        elif i == Ny-1 and j > 0 and j < Nx-1:
            transmissibilidade_x_leste = 0
            transmissibilidade_x_oeste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i-1]))
            transmissibilidade_y_sul = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j+1, i]))
            transmissibilidade_y_norte = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j-1, i]))

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx *
                                                                                     transmissibilidade_x_oeste) + (beta*ry*transmissibilidade_y_sul) + (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j, i-1])] = - \
                (beta*rx*transmissibilidade_x_oeste)
            A_[i_0, int(indice[j-1, i])] = - \
                (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j+1, i])] = -(beta*ry*transmissibilidade_y_sul)
            F[i_0] = 0

        elif j == Nx-1 and i > 0 and i < Ny-1:
            transmissibilidade_x_leste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i+1]))
            transmissibilidade_x_oeste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i-1]))
            transmissibilidade_y_sul = 0
            transmissibilidade_y_norte = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j-1, i]))

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx *
                                                                                     transmissibilidade_x_oeste) + (beta*ry*transmissibilidade_y_sul) + (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j, i+1])] = - \
                (beta*rx*transmissibilidade_x_leste)
            A_[i_0, int(indice[j, i-1])] = - \
                (beta*rx*transmissibilidade_x_oeste)
            A_[i_0, int(indice[j-1, i])] = - \
                (beta*ry*transmissibilidade_y_norte)
            F[i_0] = 0

        elif j > 0 and i > 0 and j < Nx-1 and i < Ny-1:
            transmissibilidade_x_leste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i+1]))
            transmissibilidade_x_oeste = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j, i-1]))
            transmissibilidade_y_sul = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j+1, i]))
            transmissibilidade_y_norte = 2 / \
                ((1/permeabilidades[j, i]) + (1/permeabilidades[j-1, i]))

            A_[i_0, int(indice[j, i])] = 1 + (beta*rx*transmissibilidade_x_leste) + (beta*rx *
                                                                                     transmissibilidade_x_oeste) + (beta*ry*transmissibilidade_y_sul) + (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j, i+1])] = - \
                (beta*rx*transmissibilidade_x_leste)
            A_[i_0, int(indice[j, i-1])] = - \
                (beta*rx*transmissibilidade_x_oeste)
            A_[i_0, int(indice[j-1, i])] = - \
                (beta*ry*transmissibilidade_y_norte)
            A_[i_0, int(indice[j+1, i])] = -(beta*ry*transmissibilidade_y_sul)
            F[i_0] = 0

        i_0 += 1


# ################################################################################


A_sparse = csr_matrix(A_)

P = np.ones(Nx*Ny)*P0
P_old = P.copy()

vazao_list = []          # Vazão nos tempos dos frames
producao_acumulada = 0   # Produção acumulada
producao_list = []       # Produção acumulada nos tempos dos frames
tempo_list = []          # lista de tempo

solucao_animada = []
tempos_animacao = []

t = 0
nt = int(t_max / dt)

for step in range(nt + 1):
    t = step * dt

    # Resolver sistema linear
    d = P_old + F
    P_new = spsolve(A_sparse, d)

    # Atualizar para próximo passo
    P_old = P_new

    # Calcular vazão e produção acumulada
    P_poco = P_new[int(indice[indice_poco, indice_poco])]
    qw = J * (P_poco - Pwf_prod)
    producao_acumulada += qw * dt

    tempo_list.append(t)
    vazao_list.append(qw)
    producao_list.append(producao_acumulada)
    if t % (3600) == 0:
        print(f'Tempo: {t/3600:.2f} horas simulada')
    if t % (salvamento) == 0:
        tempos_animacao.append(t)
        solucao_animada.append(P_new.copy()/kgf_cm2_to_Pa)

P_new = P_new/kgf_cm2_to_Pa

plt.figure(figsize=(6, 5))
plt.imshow(
    P_new.reshape(Nx, Ny),
    cmap='viridis',
    origin='lower',
    extent=[0, Lx, 0, Ly],
    vmin=Pwf_prod/kgf_cm2_to_Pa,
    vmax=P0/kgf_cm2_to_Pa
)
plt.colorbar(label='Pressão [kgf/cm²]')
plt.title("Distribuição de Pressão no Reservatório (Instante Final)")
title = f'Pressão no Reservatório  dt = {dt} s, Nx = {Nx}, Ny = {Ny}'
plt.savefig(f'{title}.png', dpi=300)
plt.show()
plt.close()


# plt.plot(np.array(tempo_list)/3600, producao_list, label=f'Poço produtor localizado em {indice_poco}x{indice_poco}')

# plt.xlabel('Tempo [h]')
# plt.ylabel('Produção [m³]')
# plt.title('Produção de Óleo')
# plt.grid(alpha=0.7)
# plt.legend()
# plt.savefig(f'{title}_np.png', dpi=300)
# plt.close()

# plt.plot(np.array(tempo_list)/3600, vazao_list,label=f'Poço produtor localizado em {indice_poco}x{indice_poco}')

# plt.xlabel('Tempo [h]')
# plt.ylabel('Vazão [m³/s]')
# plt.title('Vazão de Óleo')
# plt.grid(alpha=0.7)
# plt.legend()
# plt.savefig(f'{title}_qw.png', dpi=300)
# plt.close()


# fig, ax = plt.subplots()
# cax = ax.imshow(
#     solucao_animada[0].reshape(Nx, Ny),
#     cmap='viridis',
#     origin='upper',
#     extent=[0, Lx, Ly, 0],
#     vmin=Pwf_prod/kgf_cm2_to_Pa,
#     vmax=P0/kgf_cm2_to_Pa,
# )
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# ax.xaxis.set_label_position('top')
# fig.colorbar(cax, ax=ax)
# ax.set_title(f"T = 0.00 s")
# fig.tight_layout()

# def animate(frame):
#     cax.set_array(solucao_animada[frame].reshape(Nx, Ny))
#     ax.set_title(f"T = {tempos_animacao[frame]/3600:.2f} h")
#     return cax,

# ani = animation.FuncAnimation(
#     fig,
#     animate,
#     frames=len(solucao_animada),
#     interval=33,  # ~30 fps
#     blit=False  #
# )
# # Salvar animação como MP4
# writer = FFMpegWriter(fps=30, bitrate=5000)
# ani.save(f"{title}_video.mp4", writer=writer, dpi=300)

# plt.show()
# plt.close()

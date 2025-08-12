from mpl_toolkits.mplot3d import Axes3D
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import time



def expand_numbers_safe(text):
    valores = []
    for token in text.split():
        if token == "/" or token.strip() == "":
            continue
        if "*" in token:
            try:
                n, val = token.split("*")
                valores.extend([float(val)] * int(n))
            except ValueError:
                continue
        else:
            try:
                valores.append(float(token))
            except ValueError:
                continue
    return valores

# Função para extrair valores de uma keyword (ex: PERMX)


def extrair_keyword(linhas, keyword):
    valores = []
    capturando = False
    for linha in linhas:
        if keyword in linha.upper():
            capturando = True
            continue
        if capturando:
            # Se encontrar nova keyword, parar
            if re.match(r"^[A-Z]", linha.strip(), re.IGNORECASE):
                break
            valores.extend(expand_numbers_safe(linha))
    return valores


# Caminho do arquivo
file_path = "PETRO_0.INC"  # ajuste para o seu

# Ler todas as linhas
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    linhas = f.readlines()

# Extrair permeabilidades
permx = extrair_keyword(linhas, "PERMX")
permy = extrair_keyword(linhas, "PERMY")
permz = extrair_keyword(linhas, "PERMZ")

print(f"PERMX: {len(permx)} valores extraídos")
print(f"PERMY: {len(permy)} valores extraídos")
print(f"PERMZ: {len(permz)} valores extraídos")


pontos = np.column_stack((np.array(permx), np.array(permy), np.array(permz)))


def extrair_coord_numpy(arquivo):
    coords = []
    in_coord = False

    with open(arquivo, 'r') as f:
        for line in f:
            if "COORD" in line:
                in_coord = True
                continue

            if in_coord:
                if "/" in line:  # Fim da seção
                    break

                # Extrair todos os números válidos
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line.split('--')[0])
                coords.extend(map(float, numbers))

    return np.array(coords).reshape(-1, 3)


# Exemplo de uso
arquivo_data = "UNISIM_I_D_ECLIPSE.data"
coords_array = extrair_coord_numpy(arquivo_data)



def extrair_zcorn(caminho_arquivo):
    with open(caminho_arquivo, 'r') as arquivo:
        dentro_zcorn = False
        dados_zcorn = []

        for linha in arquivo:
            if 'ZCORN' in linha:
                dentro_zcorn = True
                continue

            if dentro_zcorn:
                if '/' in linha:

                    linha = linha.replace('/', '')
                    dados_zcorn.extend(map(float, linha.split()))
                    break  # fim do bloco ZCORN
                else:
                    dados_zcorn.extend(map(float, linha.split()))

    return dados_zcorn


# Exemplo de uso
caminho = 'UNISIM_I_D_ECLIPSE.data'
zcorn_dados = extrair_zcorn(caminho)


NX, NY, NZ = 81, 58, 20  # ajuste para seu modelo


def ler_actnum_arquivo(caminho_arquivo):
    valores_str = []
    coletando = False

    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            linha = linha.strip()

            if not coletando:
                if linha.upper() == "ACTNUM":
                    coletando = True
                continue

            # Quando estiver coletando, para se achar a linha que termina com '/'
            valores_str.append(linha)
            if linha.endswith('/'):
                break

    # Junta tudo e remove barras
    texto_valores = " ".join(valores_str).replace('/', ' ')

    # Separa tokens e converte para int (0 ou 1)
    tokens = texto_valores.split()
    valores = [int(t) for t in tokens if t in ('0', '1')]

    return np.array(valores)


# Exemplo de uso
array_actnum = ler_actnum_arquivo("UNISIM_I_D_ECLIPSE.data")


for i in range(NX):
    for j in range(NY):
        for k in range(NZ):
            idx = i + j*NX + k*NX*NY
            if array_actnum[idx] == 1:
                print(f"Ativo ({i}, {j}, {k})")
            else:
                print(f"Inativo ({i}, {j}, {k})")


x, y, z = coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]


min_x = np.min(x)
max_x = np.max(x)
min_y = np.min(y)
max_y = np.max(y)
min_z = np.min(z)
max_z = np.max(z)


print(f"Min X: {min_x}, Max X: {max_x}")
print(f"Min Y: {min_y}, Max Y: {max_y}")
print(f"Min Z: {min_z}, Max Z: {max_z}")


# Criar vetores de coordenadas
x_vet = np.linspace(min_x, max_x, NX)
y_vet = np.linspace(min_y, max_y, NY)
z_vet = np.linspace(min_z, max_z, NZ)

# Criar grid 3D (em ordem F para casar com ACTNUM de simuladores como Eclipse)
X, Y, Z = np.meshgrid(x_vet, y_vet, z_vet, indexing='ij')

# Transformar em colunas 1D
Xf = X.flatten(order='F')
Yf = Y.flatten(order='F')
Zf = Z.flatten(order='F')

# Selecionar apenas blocos ativos
mask_ativos = array_actnum == 1
X_ativos = Xf[mask_ativos]
Y_ativos = Yf[mask_ativos]
Z_ativos = Zf[mask_ativos]

# Plotar apenas blocos ativos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_ativos, Y_ativos, Z_ativos, s=2, c='red')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Remodela ACTNUM e permeabilidades para 3D (NX, NY, NZ)
actnum_3d = array_actnum.reshape((NX, NY, NZ), order='F')
permX_3d = np.array(permx).reshape((NX, NY, NZ), order='F')
permY_3d = np.array(permy).reshape((NX, NY, NZ), order='F')
permZ_3d = np.array(permz).reshape((NX, NY, NZ), order='F')


# Agora cada célula (i, j, k) tem seu valor
for i in range(NX):
    for j in range(NY):
        for k in range(NZ):
            if actnum_3d[i, j, k] == 1:
                status = "Ativo"
                print(f"({i}, {j}, {k}) | {status} | "
                      f"PermX={permX_3d[i, j, k]:.2f} | "
                      f"PermY={permY_3d[i, j, k]:.2f} | "
                      f"PermZ={permZ_3d[i, j, k]:.2f}")
                    

            else:
                status = "Inativo"
                if (permX_3d[i, j, k] != 1 or
                    permY_3d[i, j, k] != 1 or
                        permZ_3d[i, j, k] != 1):
                    raise ValueError(
                        f"Célula ({i}, {j}, {k}) inativa com permeabilidade ≠ 1: "
                        f"PermX={permX_3d[i, j, k]}, PermY={permY_3d[i, j, k]}, PermZ={permZ_3d[i, j, k]}"
                    )




# --- Remodelar para 3D ---
actnum_3d = array_actnum.reshape((NX, NY, NZ), order='F')
permX_3d = np.array(permx).reshape((NX, NY, NZ), order='F')
permY_3d = np.array(permy).reshape((NX, NY, NZ), order='F')
permZ_3d = np.array(permz).reshape((NX, NY, NZ), order='F')

# --- Criar o grid 3D ---
x_vet = np.linspace(min_x, max_x, NX)
y_vet = np.linspace(min_y, max_y, NY)
z_vet = np.linspace(min_z, max_z, NZ)
X, Y, Z = np.meshgrid(x_vet, y_vet, z_vet, indexing='ij')

# --- Flatten e filtrar ativos ---
ativos = actnum_3d.flatten(order='F') == 1
Xf = X.flatten(order='F')[ativos]
Yf = Y.flatten(order='F')[ativos]
Zf = Z.flatten(order='F')[ativos]

permX_flat = permX_3d.flatten(order='F')[ativos]
permY_flat = permY_3d.flatten(order='F')[ativos]
permZ_flat = permZ_3d.flatten(order='F')[ativos]

# # --- Plot com 3 subplots ---
# fig = plt.figure(figsize=(15, 5))

# PermX
# ax1 = fig.add_subplot(131, projection='3d')
# sc1 = ax1.scatter(Xf, Yf, Zf[:-1], c=permX_flat, cmap='viridis', s=5)
# ax1.set_title("PermX")
# ax1.set_xlabel("X")
# ax1.set_ylabel("Y")
# ax1.set_zlabel("Z")
# fig.colorbar(sc1, ax=ax1, shrink=0.5, label="mD")

# # PermY
# ax2 = fig.add_subplot(132, projection='3d')
# sc2 = ax2.scatter(Xf, Yf, Zf, c=permY_flat, cmap='viridis', s=5)
# ax2.set_title("PermY")
# ax2.set_xlabel("X")
# ax2.set_ylabel("Y")
# ax2.set_zlabel("Z")
# fig.colorbar(sc2, ax=ax2, shrink=0.5, label="mD")

# # PermZ
# ax3 = fig.add_subplot(133, projection='3d')
# sc3 = ax3.scatter(Xf, Yf, Zf, c=permZ_flat, cmap='viridis', s=5)
# ax3.set_title("PermZ")
# ax3.set_xlabel("X")
# ax3.set_ylabel("Y")
# ax3.set_zlabel("Z")
# fig.colorbar(sc3, ax=ax3, shrink=0.5, label="mD")

# plt.tight_layout()

# Remodelar arrays para 3D
actnum_3d = array_actnum.reshape((NX, NY, NZ), order='F')
permX_3d = np.array(permx).reshape((NX, NY, NZ), order='F')

# Coordenadas
x_vet = np.linspace(min_x, max_x, NX)
y_vet = np.linspace(min_y, max_y, NY)
X2d, Y2d = np.meshgrid(x_vet, y_vet, indexing='ij')
Xf = X2d.flatten(order='F')
Yf = Y2d.flatten(order='F')

# Colormap customizado
cores = ["#ff00b3", "#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"]
cmap_custom = LinearSegmentedColormap.from_list("CustomMap", cores)


for k in range(NZ):
  
    permX_k = permX_3d[:, :, k].flatten(order='F')
    actnum_k = actnum_3d[:, :, k].flatten(order='F')

 
    mask_ativos = actnum_k == 1
    Xf_ativos = Xf[mask_ativos]
    Yf_ativos = Yf[mask_ativos]
    permX_ativos = permX_k[mask_ativos]


    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Xf_ativos, Yf_ativos, c=permX_ativos, cmap=cmap_custom, s=20)
    plt.colorbar(sc, label='PermX (mD)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(min_x-1000, max_x+1000)
    plt.ylim(min_y-1000, max_y+1000)
    plt.title(f'PermX camada k={k} - Blocos ativos')
    plt.gca().set_aspect('equal')
    plt.grid(alpha=0.2)
    plt.show()
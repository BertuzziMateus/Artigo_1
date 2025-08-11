import re
import matplotlib.pyplot as plt
import numpy as np

# Função para expandir formato tipo "53*0.00" em lista de floats


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



import re
import numpy as np

def extrair_coord_numpy(arquivo):
    """
    Lê um arquivo .data e retorna um NumPy array (N x 3) com as coordenadas X, Y, Z
    da seção COORD.
    """
    coords = []
    lendo_coord = False
    
    with open(arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            linha = linha.strip()
            
            # Início da seção COORD
            if linha.startswith("COORD"):
                lendo_coord = True
                continue
            
            # Fim da seção
            if lendo_coord and linha.startswith("/"):
                break
            
            if lendo_coord:
                # Remove comentários '--'
                linha_limpa = linha.split('--')[0]
                # Extrai números (float ou int)
                numeros = re.findall(r"[-+]?\d*\.\d+|\d+", linha_limpa)
                coords.extend(map(float, numeros))
    
    # Converte para NumPy e reestrutura em colunas X, Y, Z
    return np.array(coords).reshape(-1, 3)

# Exemplo de uso
arquivo_data = "UNISIM_I_D_ECLIPSE.data"
coords_array = extrair_coord_numpy(arquivo_data)

# Salvando para uso posterior
np.save("coordenadas.npy", coords_array)

# Plotando exemplo 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], s=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()






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
                    # Remove a barra e adiciona os últimos valores
                    linha = linha.replace('/', '')
                    dados_zcorn.extend(map(float, linha.split()))
                    break  # fim do bloco ZCORN
                else:
                    dados_zcorn.extend(map(float, linha.split()))

    return dados_zcorn

# Exemplo de uso
caminho = 'UNISIM_I_D_ECLIPSE.data'
zcorn_dados = extrair_zcorn(caminho)
print(f"Número de valores extraídos: {len(zcorn_dados)}")


#print(zcorn_dados)


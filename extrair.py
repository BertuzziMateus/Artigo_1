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
file_path = "PETRO_0[1].INC"  # ajuste para o seu

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

#print("PERMX primeiros:", permx)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Magnitude total da permeabilidade (opcional, para colorir)
#magnitude = np.sqrt(permx**2 + permy**2 + permz**2)

p = ax.scatter(permx, permy, permz, cmap='viridis', s=2)
fig.colorbar(p, label='Magnitude da Permeabilidade (mD)')

ax.set_xlabel('PermX')
ax.set_ylabel('PermY')
ax.set_zlabel('PermZ')
plt.show()
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

def extrair_coord(arquivo):
    """
    Lê um arquivo .data e retorna uma lista com as coordenadas (X, Y, Z)
    encontradas na seção COORD.
    """
    coords = []
    lendo_coord = False
    
    with open(arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            linha = linha.strip()
            
            # Detecta início da seção COORD
            if linha.startswith("COORD"):
                lendo_coord = True
                continue
            
            # Detecta fim da seção
            if lendo_coord and linha.startswith("/"):
                break
            
            if lendo_coord:
                # Remove comentários '--' e transforma a linha em números
                linha_limpa = linha.split('--')[0]
                numeros = re.findall(r"[-+]?\d*\.\d+|\d+", linha_limpa)
                coords.extend(map(float, numeros))
    
    # Agrupa de 3 em 3 para formar (X, Y, Z)
    coordenadas_agrupadas = [tuple(coords[i:i+3]) for i in range(0, len(coords), 3)]
    return coordenadas_agrupadas

# Exemplo de uso
arquivo_data = "UNISIM_I_D_ECLIPSE.data"
resultado = extrair_coord(arquivo_data)
for i, (x, y, z) in enumerate(resultado, start=1):
    print(f"Bloco {i}: X={x}, Y={y}, Z={z}")







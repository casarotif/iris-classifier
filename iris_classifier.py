import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Criando nosso próprio dataset de dragões
dragoes = {
    'comprimento_corpo': [
        24, 20, 18, 25, 15, 30, 22, 12, 28, 16,
        21, 19, 26, 14, 32, 23, 11, 27, 17, 33
    ],
    'envergadura_asas': [
        19, 15, 12, 20, 10, 25, 18, 8, 22, 11,
        16, 13, 21, 9, 26, 19, 7, 23, 12, 27
    ],
    'temperatura_chama': [
        1100, 800, 600, 1000, 500, 1200, 900, 400, 1100, 550,
        850, 650, 1050, 450, 1250, 950, 350, 1150, 600, 1280
     ],
    'especie': [
        'Focinho Curto Sueco', 'Rabo-Córneo Húngaro', 'Verde-Galês', 'Meteoro Chinês', 'Verde-Galês', 
        'Rabo-Córneo Húngaro', 'Focinho Curto Sueco', 'Meteoro Chinês', 'Verde-Galês', 'Rabo-Córneo Húngaro',
        'Verde-Galês', 'Meteoro Chinês', 'Focinho Curto Sueco', 'Rabo-Córneo Húngaro', 'Verde-Galês',
        'Meteoro Chinês', 'Verde-Galês', 'Rabo-Córneo Húngaro', 'Focinho Curto Sueco', 'Meteoro Chinês'
    ]
}

# Criando o DataFrame
df = pd.DataFrame(dragoes)

# Preparando os dados
X = df[['comprimento_corpo', 'envergadura_asas', 'temperatura_chama']].values
y = df['especie'].values

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Duplicar os dados de cada espécie
X_train_expanded = np.repeat(X_train, 3, axis=0)
y_train_expanded = np.repeat(y_train, 3)

# Criando e treinando o modelo
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train_expanded, y_train_expanded)

# Testando o modelo
precisao = modelo.score(X_test, y_test)
print(f"Precisão do modelo: {precisao:.2f}")

# Testando com um novo dragão
novo_dragao = [[20, 12, 1200]]  # comprimento, envergadura, temperatura
predicao = modelo.predict(novo_dragao)
probabilidades = modelo.predict_proba(novo_dragao)

print(f"\nPara um dragão com:")
print(f"Comprimento: {novo_dragao[0][0]} metros")
print(f"Envergadura: {novo_dragao[0][1]} metros")
print(f"Temperatura da chama: {novo_dragao[0][2]}°C")
print(f"\nO modelo prevê que é um: {predicao[0]}")

#especies = modelo.classes_
#print(especies)

# Mostrando probabilidades para cada espécie
#especies = modelo.classes_
#print("\nProbabilidades para cada espécie:")
#for especie, prob in zip(especies, probabilidades[0]):
    #    print(f"{especie}: {prob:.2%}")

#    plt.figure(figsize=(10, 6))
#sns.scatterplot(data=df, x='comprimento_corpo', y='temperatura_chama', hue='especie', style='especie', s=100)
#plt.title('Dragões por Comprimento e Temperatura da Chama')
#plt.xlabel('Comprimento do Corpo (metros)')
#plt.ylabel('Temperatura da Chama (°C)')
#plt.show()
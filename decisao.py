
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
# Dados de exemplo (12 linhas fictícias)
dados = [
    {"Alternativa": 0, "Bar": 0, "SexSab": 0, "Faminto": 1, "Clientes": 1, "Preco": 2, "Chovendo": 1, "Reserva": 0, "Tipo": 0, "Espera": 0, "Comer": 1},
    {"Alternativa": 1, "Bar": 0, "SexSab": 0, "Faminto": 0, "Clientes": 2, "Preco": 2, "Chovendo": 1, "Reserva": 1, "Tipo": 2, "Espera": 3, "Comer": 0},
    {"Alternativa": 0, "Bar": 1, "SexSab": 0, "Faminto": 1, "Clientes": 2, "Preco": 0, "Chovendo": 0, "Reserva": 0, "Tipo": 1, "Espera": 1, "Comer": 1},
    {"Alternativa": 1, "Bar": 1, "SexSab": 1, "Faminto": 1, "Clientes": 2, "Preco": 3, "Chovendo": 0, "Reserva": 1, "Tipo": 3, "Espera": 2, "Comer": 0},
    {"Alternativa": 0, "Bar": 0, "SexSab": 0, "Faminto": 1, "Clientes": 0, "Preco": 0, "Chovendo": 1, "Reserva": 0, "Tipo": 1, "Espera": 0, "Comer": 1},
    {"Alternativa": 1, "Bar": 1, "SexSab": 1, "Faminto": 1, "Clientes": 2, "Preco": 2, "Chovendo": 0, "Reserva": 0, "Tipo": 2, "Espera": 1, "Comer": 1},
    {"Alternativa": 0, "Bar": 0, "SexSab": 1, "Faminto": 0, "Clientes": 1, "Preco": 1, "Chovendo": 1, "Reserva": 1, "Tipo": 0, "Espera": 2, "Comer": 1},
    {"Alternativa": 1, "Bar": 1, "SexSab": 0, "Faminto": 0, "Clientes": 1, "Preco": 1, "Chovendo": 0, "Reserva": 0, "Tipo": 1, "Espera": 0, "Comer": 1},
    {"Alternativa": 1, "Bar": 0, "SexSab": 1, "Faminto": 1, "Clientes": 2, "Preco": 2, "Chovendo": 1, "Reserva": 0, "Tipo": 3, "Espera": 3, "Comer": 0},
    {"Alternativa": 0, "Bar": 0, "SexSab": 0, "Faminto": 1, "Clientes": 0, "Preco": 0, "Chovendo": 0, "Reserva": 0, "Tipo": 3, "Espera": 0, "Comer": 1},
    {"Alternativa": 1, "Bar": 1, "SexSab": 1, "Faminto": 0, "Clientes": 1, "Preco": 3, "Chovendo": 1, "Reserva": 1, "Tipo": 0, "Espera": 2, "Comer": 0},
    {"Alternativa": 0, "Bar": 1, "SexSab": 1, "Faminto": 1, "Clientes": 2, "Preco": 2, "Chovendo": 0, "Reserva": 1, "Tipo": 2, "Espera": 1, "Comer": 1}
]
#Convertendo para DataFrame
df = pd.DataFrame(dados)
#Separando entradas e saídas
x = df.drop("Comer", axis=1)
y=df["Comer"]
#Treinando a árvore de decisão
modelo= DecisionTreeClassifier()
modelo.fit(x, y)
#Exibindo a árvore em formato de texto
arvore = export_text(modelo, feature_names=list(x.columns))
print(arvore)

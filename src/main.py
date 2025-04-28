"""
Arquivo principal do projeto de Previsão de demanda.
Organiza a execução dos módulos da Pipeline.

Autor: Marcos Scatolino
Data: 24/04/2025
"""

# ================== IMPORTS ================== #
from preprocessing import preparar_dados
from feature_engineering import adicionar_features
from eda import executar_eda
from modeling import treinar_modelos


# ================== MAIN ================== #
if __name__ == "__main__":
    # Prepara e organiza os dados.
    df_enriquecido = preparar_dados()
    
    # Seleciona os dados e produz os gráficos para a análise
    df_features = adicionar_features(df_enriquecido)
    executar_eda(df_features)
    
    # Modelagem preditiva de demanda
    modelo_final = treinar_modelos(df_features)


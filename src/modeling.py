"""
Módulo de modelagem preditiva para previsão de demanda.

Aplica modelos de Regressão Linear e Random Forest,
com avaliação por métricas e gráficos de comparação.

Autor: Marcos Scatolino
Data: 2025-04-24
"""

# ================== IMPORTS ================== #
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os


# ================== FUNÇÃO PRINCIPAL ================== #
def treinar_modelos(df):
    """
    Prepara os dados, treina modelos e avalia o desempenho.
    Retorna o melhor modelo treinado.
    """

    # ================== FEATURES E TARGET ================== #
    features = [
        "day_of_week", "week_of_year", "month", "year",
        "is_weekend", "is_month_start", "is_month_end", "promotion", "feriado"
    ]
    target = "sales"

    # Transformações de booleanos para inteiros
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_month_start"] = df["is_month_start"].astype(int)
    df["is_month_end"] = df["is_month_end"].astype(int)
    df["feriado"] = df["feriado"].astype(int)

    # ================== DIVISÃO TREINO/TESTE ================== #
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ================== MODELO BASE: REGRESSÃO LINEAR ================== #
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # ================== RANDOM FOREST ================== #
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # ================== AVALIAÇÃO ================== #
    def avaliar_modelo(nome, y_true, y_pred):
        print(f"\n📊 Resultados do modelo: {nome}")
        print("MAE:", mean_absolute_error(y_true, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
        print("R²:", r2_score(y_true, y_pred))

    avaliar_modelo("Regressão Linear", y_test, y_pred_lr)
    avaliar_modelo("Random Forest", y_test, y_pred_rf)

    # ================== GRÁFICO COMPARATIVO ================== #
    plt.figure(num="📉 Comparação Real vs Previsto", figsize=(10, 5))
    plt.plot(y_test.values[:100], label="Real", marker="o")
    plt.plot(y_pred_rf[:100], label="Previsto RF", linestyle="--")
    plt.title("🔮 Previsão vs Real (Random Forest)")
    plt.xlabel("Observações")
    plt.ylabel("Vendas")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== SALVAR MODELO ================== #

    # Caminho absoluto até a raiz do projeto
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    modelo_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(modelo_dir, exist_ok=True)

    modelo_path = os.path.join(modelo_dir, "modelo_random_forest.pkl")
    joblib.dump(rf, modelo_path)

    print(f"\n💾 Modelo salvo com sucesso em: {modelo_path}")

    return rf

"""
main.py
Pipeline completo: carregamento → pré-processamento → treino → avaliação → relatório.
"""

import sys
import os

# Garante que src/ seja encontrado ao rodar da raiz do projeto
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, basic_info
from src.preprocessing import feature_engineering, split_data
from src.train import train_all
from src.evaluate import evaluate_all, plot_report


def main():
    print("=" * 60)
    print("   🏠 House Price Predictor — Pipeline Principal")
    print("=" * 60)

    # 1. Carregar dados
    df = load_data(save_csv=True)
    basic_info(df)

    # 2. Feature engineering
    df = feature_engineering(df)

    # 3. Divisão treino / teste
    X_train, X_test, y_train, y_test = split_data(df, target="Price")

    # 4. Treinar modelos
    trained_models = train_all(X_train, y_train)

    # 5. Avaliar
    metrics_df = evaluate_all(trained_models, X_test, y_test)

    # 6. Gerar relatório visual
    plot_report(trained_models, X_test, y_test, metrics_df)

    print("\n✅ Pipeline concluído com sucesso!")
    print("   → Dados    : data/housing.csv")
    print("   → Modelos  : models/*.pkl")
    print("   → Relatório: reports/model_report.png")


if __name__ == "__main__":
    main()

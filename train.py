"""
train.py
Treinamento e serialização dos modelos de regressão.
"""

import os
import time
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODELS = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
}


def train_all(X_train, y_train, models_dir: str = "models") -> dict:
    """
    Treina todos os modelos definidos em MODELS.

    Args:
        X_train: Features de treino
        y_train: Alvo de treino
        models_dir: Diretório para salvar os modelos

    Returns:
        Dicionário {nome: modelo treinado}
    """
    os.makedirs(models_dir, exist_ok=True)
    trained = {}

    print("\n🤖 Treinando modelos...")
    print("=" * 50)

    for name, model in MODELS.items():
        print(f"  ⏳ {name}...", end=" ", flush=True)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"✅ ({elapsed:.1f}s)")

        # Salvar modelo
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(models_dir, f"{safe_name}.pkl")
        joblib.dump(model, path)

        trained[name] = model

    print(f"\n💾 Modelos salvos em '{models_dir}/'")
    return trained


def load_model(name: str, models_dir: str = "models"):
    """
    Carrega um modelo serializado.

    Args:
        name: Nome do arquivo sem extensão (ex: 'random_forest')
        models_dir: Diretório dos modelos

    Returns:
        Modelo carregado
    """
    path = os.path.join(models_dir, f"{name}.pkl")
    return joblib.load(path)

"""
evaluate.py
Avaliação dos modelos e geração de relatórios visuais.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def compute_metrics(model, X_test, y_test) -> dict:
    """Calcula MAE, RMSE e R² para um modelo."""
    y_pred = model.predict(X_test)
    return {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":   r2_score(y_test, y_pred),
    }


def evaluate_all(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Avalia todos os modelos e retorna um DataFrame de resultados.

    Args:
        trained_models: {nome: modelo}
        X_test, y_test: Conjunto de teste

    Returns:
        DataFrame com métricas por modelo
    """
    print("\n📊 Resultados dos Modelos")
    print("=" * 55)
    print(f"{'Modelo':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print("-" * 55)

    results = []
    for name, model in trained_models.items():
        metrics = compute_metrics(model, X_test, y_test)
        results.append({"Model": name, **metrics})
        print(f"{name:<25} ${metrics['MAE']:>9,.0f} ${metrics['RMSE']:>9,.0f} {metrics['R2']:>7.4f}")

    print("=" * 55)
    df = pd.DataFrame(results).sort_values("R2", ascending=False)
    best = df.iloc[0]["Model"]
    print(f"\n🏆 Melhor modelo: {best} (R² = {df.iloc[0]['R2']:.4f})")
    return df


def plot_report(trained_models: dict, X_test, y_test,
                metrics_df: pd.DataFrame, reports_dir: str = "reports") -> None:
    """
    Gera e salva um relatório visual completo dos modelos.
    """
    os.makedirs(reports_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("House Price Predictor — Model Report", fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    model_names = list(trained_models.keys())
    colors = COLORS[:len(model_names)]

    # ── 1. Barras de R² ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(metrics_df["Model"], metrics_df["R2"], color=colors)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("R²")
    ax1.set_title("R² Score por Modelo")
    for bar, val in zip(bars, metrics_df["R2"]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)

    # ── 2. MAE e RMSE agrupados ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(metrics_df))
    w = 0.35
    ax2.bar(x - w/2, metrics_df["MAE"] / 1000, w, label="MAE", color="#4C72B0")
    ax2.bar(x + w/2, metrics_df["RMSE"] / 1000, w, label="RMSE", color="#DD8452")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split()[0] for m in metrics_df["Model"]], rotation=15)
    ax2.set_ylabel("Erro ($ mil)")
    ax2.set_title("MAE vs RMSE por Modelo")
    ax2.legend()

    # ── 3. Predito vs Real — melhor modelo ───────────────────────────────────
    best_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    y_pred_best = best_model.predict(X_test)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_test / 1000, y_pred_best / 1000, alpha=0.3, s=10, color="#55A868")
    lims = [min(y_test.min(), y_pred_best.min()) / 1000,
            max(y_test.max(), y_pred_best.max()) / 1000]
    ax3.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    ax3.set_xlabel("Valor Real ($ mil)")
    ax3.set_ylabel("Valor Predito ($ mil)")
    ax3.set_title(f"Predito vs Real\n({best_name})")
    ax3.legend()

    # ── 4. Distribuição dos resíduos — melhor modelo ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    residuals = y_test.values - y_pred_best
    ax4.hist(residuals / 1000, bins=50, color="#C44E52", edgecolor="white", linewidth=0.3)
    ax4.axvline(0, color="black", linestyle="--", linewidth=1)
    ax4.set_xlabel("Resíduo ($ mil)")
    ax4.set_ylabel("Frequência")
    ax4.set_title(f"Distribuição dos Resíduos\n({best_name})")

    # ── 5. Feature importances — Random Forest ───────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    rf_model = trained_models.get("Random Forest")
    if rf_model is not None:
        importances = rf_model.feature_importances_
        feat_names = X_test.columns
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=True)
        ax5.barh(fi_df["Feature"], fi_df["Importance"], color="#4C72B0")
        ax5.set_xlabel("Importância")
        ax5.set_title("Feature Importances (Random Forest)")

    path = os.path.join(reports_dir, "model_report.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📈 Relatório salvo em '{path}'")

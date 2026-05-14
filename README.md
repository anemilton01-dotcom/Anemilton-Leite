# 🏠 House Price Predictor

Um projeto de Machine Learning para prever preços de imóveis usando técnicas de regressão e análise exploratória de dados (EDA).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Sobre o Projeto

Este projeto utiliza o clássico **California Housing Dataset** para construir e comparar modelos de regressão capazes de prever o valor médio de imóveis com base em características como localização, tamanho, renda da região, entre outros.

### Objetivos
- Realizar uma análise exploratória completa dos dados
- Aplicar técnicas de pré-processamento e feature engineering
- Treinar e comparar múltiplos modelos de regressão
- Avaliar os modelos com métricas robustas
- Gerar relatório visual com os resultados

---

## 🗂️ Estrutura do Projeto

```
ml-price-predictor/
│
├── data/                  # Dados brutos e processados
│   └── housing.csv
│
├── notebooks/             # Jupyter Notebooks de exploração
│   └── eda.ipynb
│
├── src/                   # Código-fonte principal
│   ├── __init__.py
│   ├── data_loader.py     # Carregamento e limpeza dos dados
│   ├── preprocessing.py   # Pré-processamento e feature engineering
│   ├── train.py           # Treinamento dos modelos
│   └── evaluate.py        # Avaliação e métricas
│
├── models/                # Modelos serializados (.pkl)
│
├── reports/               # Gráficos e relatórios gerados
│
├── main.py                # Pipeline principal
├── requirements.txt
└── README.md
```

---

## 🚀 Como Usar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/ml-price-predictor.git
cd ml-price-predictor
```

### 2. Crie o ambiente virtual e instale as dependências

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Execute o pipeline completo

```bash
python main.py
```

### 4. (Opcional) Explore o notebook de EDA

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 📊 Modelos Utilizados

| Modelo                  | Descrição                                      |
|-------------------------|------------------------------------------------|
| Linear Regression       | Modelo base simples                            |
| Ridge Regression        | Regressão com regularização L2                 |
| Random Forest Regressor | Ensemble de árvores de decisão                 |
| Gradient Boosting       | Boosting com árvores de decisão                |

---

## 📈 Métricas de Avaliação

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R²** – Coeficiente de determinação

---

## 🛠️ Tecnologias

- Python 3.9+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib
- jupyter

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

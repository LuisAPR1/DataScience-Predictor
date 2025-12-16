# Flight Cancellation Prediction App

Uma aplicação web moderna para prever o cancelamento de voos, utilizando modelos de Machine Learning avançados através de uma pipeline de dados rigorosa.

## 🚀 Funcionalidades
*   **Previsão Única:** Preencha os dados do voo e obtenha instantaneamente a probabilidade de cancelamento.
*   **Avaliação em Lote:** Faça upload de um CSV com histórico e avalie a performance (Accuracy, F1, Recall, Precision) real dos modelos.
*   **Multi-Modelos:** Escolha entre Naive Bayes, KNN, Logistic Regression, Decision Trees, MLP e Random Forest.
*   **Pipeline Robusta:** Tratamento automático de valores em falta, codificação cíclica temporal e encoding categórico.

## 🛠️ Tecnologias
*   **Backend:** Python, FastAPI, Scikit-Learn, Pandas, Joblib.
*   **Frontend:** HTML5, Vanilla JS, CSS3 (Glassmorphism/Dark Theme).

## 📦 Instalação e Execução

### Pré-requisitos
*   Python 3.8+
*   Virtual Environment (Recomendado)

### 1. Configurar Ambiente
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar (Windows)
.venv\Scripts\activate

# Instalar dependências
pip install fastapi "uvicorn[standard]" pandas scikit-learn numpy joblib python-multipart
```

### 2. Treinar Modelos (Opcional)
Se precisar de regerar os modelos ou se tiver novos dados:
```bash
python save_objects.py
```
*Isto irá ler os dados de `datasets/`, treinar os 6 modelos e guardar os ficheiros `.joblib` na pasta `models/`.*

### 3. Iniciar Servidor
```bash
python -m uvicorn main:app --reload
```

### 4. Usar a App
Abra o browser em: `http://127.0.0.1:8000`

## 📂 Estrutura do Projeto
```
├── main.py                 # Servidor Web (API FastAPI)
├── pipeline.py             # Lógica de transformação de dados
├── save_objects.py         # Script de treino e persistência de modelos
├── prediction_objects.json # Manifesto com definição da pipeline e modelos
├── static/                 # Frontend (HTML, CSS, JS)
├── models/                 # Modelos treinados e encoders (.joblib)
├── datasets/               # Ficheiros de dados (não incluídos no git)
└── codes/                  # Scripts originais de preprocessing (referência)
```

## ⚠️ Notas
*   **Modelos Recomendados:** Naive Bayes e KNN têm melhor desempenho na detecção de voos cancelados.
*   Certifique-se que usa o mesmo ambiente Python para treinar (`save_objects.py`) e para correr o servidor (`main.py`) para evitar avisos de versão do `scikit-learn`.


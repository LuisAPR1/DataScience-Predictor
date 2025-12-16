# Resumo do Projeto: Flight Prediction System

## 1. O que é suposto funcionar (Objetivos)
O objetivo principal deste projeto é disponibilizar uma interface web para prever o cancelamento de voos com base em dados históricos, utilizando uma pipeline de Machine Learning robusta e fiel ao processo de treino original.

**Requisitos Chave:**
*   **Pipeline Fiel:** A transformação de dados na inferência (Web) deve ser **exatamente igual** à usada no treino dos modelos (Notebooks).
*   **Modelos Variados:** Suporte para múltiplos algoritmos de classificação (Naive Bayes, KNN, Logistic Regression, Decision Trees, MLP, Random Forest).
*   **Inferência Unitária:** Permitir ao utilizador preencher um formulário com todos os dados de um voo e receber uma previsão (Cancelado/Não Cancelado).
*   **Avaliação (Performance Estimation):** Permitir o upload de um ficheiro CSV com novos dados "crus" para testar a performance dos modelos num cenário real "end-to-end".
*   **Interface Profissional:** Um design moderno (Dark Theme/Glassmorphism) e tratamento de erros robusto.

## 2. O que realmente faz (Implementação Atual)
O sistema foi implementado com sucesso utilizando **FastAPI** (Backend) e **JavaScript/HTML5** (Frontend), cumprindo todos os requisitos propostos.

### Backend (`main.py` + `pipeline.py`)
*   **API REST:** Endpoints para listar modelos, prever um voo (`/predict-single`) e avaliar ficheiros (`/evaluate-models`).
*   **Pipeline Class (`PredictionPipeline`):**
    *   **Carregamento de Artefactos:** Carrega automaticamente Encoders, Scalers e Modelos `.joblib` treinados.
    *   **Missing Value Imputation (MVI):** Preenche automaticamente campos vazios com 0 na inferência para evitar erros.
    *   **Enforce Types:** Garante que números vêm como números e texto como texto, prevenindo erros de tipagem.
    *   **Ciclos Temporais:** Transforma `Month`, `DayOfWeek`, `Times` em componentes Seno/Cosseno, preservando a ciclicidade.
    *   **Ordinal Encoding:** Converte categorias (Airline, City, etc.) usando o mesmo mapeamento do treino.
    *   **Scaling:** Aplica MinMax Scaling ajustado aos dados de treino.
    *   **Seleção de Features:** Filtra exatamente as colunas que o modelo espera.

### Frontend (`index.html`, `script.js`, `style.css`)
*   **Design Premium:** Interface escura com painéis translúcidos, inspirada em "Digit Recognition Neural Network".
*   **Formulário Completo:** Inputs para todas as features necessárias, ordenados e agrupados logicamente.
*   **Feedback em Tempo Real:** Mostra visualmente se a previsão é segura (Verde) ou Cancelada (Vermelho).
*   **Gestão de Erros:**
    *   Valida uploads de ficheiros (tipo, conteúdo vazio).
    *   Exibe caixas de erro vermelhas com mensagens claras vindas do servidor (ex: "Invalid file format").
    *   Lida com falhas de API sem bloquear a página.

### Treino (`save_objects.py`)
*   Script autónomo que lê os dados de treino (`flights_best_fs_train.csv`), treina todos os 6 modelos e guarda-os como objetos persistentes prontos a usar pelo Backend.

---
**Conclusão:** O projeto é uma tradução fiel e operacional de uma experiência de Data Science para uma Aplicação Web funcional e pronta a ser utilizada por um utilizador final.

# Nestlé B2B Economic Moat 

Aplicação interativa e offline em **Marimo** voltada para análise de retenção (*Economic Moat*), prevenção de churn, prescrição de *Next Best Action* para Customer Success (CS) e fechamento de ciclo via **Data Flywheel** local em operações B2B (ex.: canais *Padaria*, *Confeitaria*, *Cafeteria*, *Mercadinho* e *Restaurante/Delivery*).

## Visão Geral da Solução

Em operações B2B, reter contas não depende apenas de preço — passa por profundidade de relacionamento, adesão a ativos/comodato, uso de crédito, amplitude de mix (*SKUs*) e múltiplos pontos de contato. Este notebook reúne:

1. **Ingestão resiliente com fallback:** carrega arquivo enviado pelo usuário (`.csv` / `.xlsx`), detecta base local (`base_historica_nestle_moat_5000.csv`) ou gera uma base sintética estruturada de 5.000 PDVs na memória.
2. **Modelo preditivo de risco (`GradientBoostingClassifier`):** treina com divisão estratificada, avalia `ROC-AUC` e calcula o **Moat Score** (`(1 - prob_churn) * 100`, de `0` a `100`).
3. **Classificação de fortaleza e *Next Best Action*:**
   - **Castelo Blindado (`score >= 75`):** foco em *upsell / true-up* ou renovação plurianual.
   - **Em Consolidação (`45 <= score < 75`):** foco em *multi-threading*, comodato, selo de *co-branding* ou *cross-sell*.
   - **Zona de Risco (`score < 45`):** foco em auditoria de *pull-through* em comodato, blindagem financeira/crédito ou intervenção urgente de CS.
4. **Priorização composta (`Índice CS`):** combina **65% de risco de churn** (`1 - moat_score / 100`) e **35% de MRR normalizado** para ordenar a fila de atuação e exportar `lista_acao_cs_moat.csv`.
5. **Data Flywheel local:** registra desfechos de intervenções de CS (`Sucesso`, `Em Negociacao`, `Sem Sucesso`) em `flywheel_feedback_log.csv` e permite recalibrar o modelo sob demanda via botão de retreinamento.

## Features Utilizadas no Modelo

- `comodato_maquina`: presença de equipamento em comodato no PDV.
- `pull_through_ratio`: razão de consumo/aderência associada ao comodato.
- `selo_cobranding`: uso de selo oficial de marca parceira no ponto de venda.
- `uso_credito_b2b`: nível de utilização da linha de crédito B2B (`0` a `1`).
- `skus_distintos`: variedade de itens/SKUs ativos na conta.
- `pedidos_app_ratio`: proporção de pedidos realizados via canal digital/app.
- `reajustes_absorvidos`: histórico de repasses/reajustes absorvidos pela conta.
- `multi_threading_users`: quantidade de usuários ativos/engajados na conta.

## Como Executar

```bash
pip install marimo pandas numpy scikit-learn openpyxl
marimo edit Moat_Nestle_Marimo_Completo.py

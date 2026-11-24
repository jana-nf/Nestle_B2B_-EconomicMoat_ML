import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="Nestle B2B - Economic Moat & Data Flywheel (100% Offline)",
    auto_download=["html"],
)


@app.cell
def _():
    import io
    import os
    from datetime import datetime
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    return (
        GradientBoostingClassifier,
        datetime,
        io,
        mo,
        np,
        os,
        pd,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # Radar de Economic Moat e Data Flywheel B2B
    **Modelo Preditivo e Prescritivo — Estudo de Caso Nestle B2B**
    *Sem APIs e sem servidores externos.*
    """)
    return


@app.cell
def _(mo):
    arquivo_upload = mo.ui.file(
        filetypes=[".csv", ".xlsx"],
        label="Opcional: Subir planilha real do ERP/CRM (.csv ou .xlsx)",
    )
    btn_retreinar = mo.ui.run_button(
        label="Retreinar Modelo com Logs do Data Flywheel"
    )
    canal_filtro = mo.ui.dropdown(
        options=[
            "Todos",
            "Padaria",
            "Confeitaria",
            "Mercadinho",
            "Cafeteria",
            "Restaurante/Delivery",
        ],
        value="Todos",
        label="Filtrar Canal B2B:",
    )
    mo.hstack([arquivo_upload, canal_filtro, btn_retreinar], justify="start", gap=2)
    return arquivo_upload, btn_retreinar, canal_filtro


@app.cell
def _(arquivo_upload, io, np, os, pd):
    FEATURES = [
        "comodato_maquina",
        "pull_through_ratio",
        "selo_cobranding",
        "uso_credito_b2b",
        "skus_distintos",
        "pedidos_app_ratio",
        "reajustes_absorvidos",
        "multi_threading_users",
    ]
    LOG_PATH = "flywheel_feedback_log.csv"

    def gerar_base_sintetica(n=5000, seed=42):
        rng = np.random.default_rng(seed)
        canais = ["Padaria", "Confeitaria", "Mercadinho", "Cafeteria", "Restaurante/Delivery"]
        regioes = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]

        canal = rng.choice(canais, size=n, p=[0.28, 0.22, 0.25, 0.15, 0.10])
        regiao = rng.choice(regioes, size=n, p=[0.45, 0.20, 0.18, 0.10, 0.07])
        comodato = rng.binomial(1, 0.38, size=n)
        pull_through = np.where(
            comodato == 1,
            np.clip(rng.normal(0.95, 0.25, size=n), 0.1, 1.8),
            0.0,
        )
        selo = rng.binomial(1, 0.34, size=n)
        credito = np.clip(rng.beta(2.2, 2.8, size=n), 0.0, 1.0)
        skus = rng.integers(1, 16, size=n)
        pedidos_app = np.clip(rng.beta(2.5, 2.0, size=n), 0.0, 1.0)
        reajustes = rng.integers(0, 4, size=n)
        usuarios = rng.integers(1, 7, size=n)
        mrr = np.round(
            900
            + skus * 210
            + comodato * 1450 * np.maximum(pull_through, 0.5)
            + selo * 650
            + rng.normal(0, 350, size=n),
            2,
        )
        mrr = np.maximum(mrr, 650.0)

        z = (
            -1.45
            + 1.35 * comodato
            + 0.95 * pull_through
            + 1.15 * selo
            + 1.10 * credito
            + 0.14 * skus
            + 0.85 * pedidos_app
            + 0.55 * reajustes
            + 0.38 * usuarios
            + rng.normal(0, 0.45, size=n)
        )
        prob_retencao = 1 / (1 + np.exp(-z))
        churn_event = rng.binomial(1, 1 - prob_retencao)

        return pd.DataFrame(
            {
                "pdv_id": [f"PDV-{i:05d}" for i in range(1, n + 1)],
                "canal": canal,
                "regiao": regiao,
                "comodato_maquina": comodato,
                "pull_through_ratio": np.round(pull_through, 2),
                "selo_cobranding": selo,
                "uso_credito_b2b": np.round(credito, 2),
                "skus_distintos": skus,
                "pedidos_app_ratio": np.round(pedidos_app, 2),
                "reajustes_absorvidos": reajustes,
                "multi_threading_users": usuarios,
                "mrr_atual_brl": mrr,
                "churn_event": churn_event,
            }
        )

    if arquivo_upload.value:
        nome_arq = arquivo_upload.name()
        conteudo = arquivo_upload.contents()
        if nome_arq.endswith(".xlsx"):
            df_base = pd.read_excel(io.BytesIO(conteudo))
        else:
            df_base = pd.read_csv(io.BytesIO(conteudo))
        origem_dados = f"Arquivo carregado: {nome_arq} ({len(df_base)} registros)"
    elif os.path.exists("base_historica_nestle_moat_5000.csv"):
        df_base = pd.read_csv("base_historica_nestle_moat_5000.csv")
        origem_dados = f"Base local detectada ({len(df_base)} registros)"
    else:
        df_base = gerar_base_sintetica(5000)
        origem_dados = "Base sintetica automatica gerada na memoria (5.000 PDVs Nestle)"
    return FEATURES, LOG_PATH, df_base, origem_dados


@app.cell
def _(
    FEATURES,
    GradientBoostingClassifier,
    LOG_PATH,
    btn_retreinar,
    df_base,
    os,
    pd,
    roc_auc_score,
    train_test_split,
):
    _ = btn_retreinar.value

    df_treino = df_base.copy()
    logs_incorporados = 0

    if os.path.exists(LOG_PATH):
        df_logs = pd.read_csv(LOG_PATH)
        df_validos = df_logs[df_logs["desfecho"].isin(["Sucesso", "Sem Sucesso"])].copy()
        if not df_validos.empty:
            df_validos["churn_event"] = (df_validos["desfecho"] == "Sem Sucesso").astype(int)
            cols_comuns = [c for c in df_treino.columns if c in df_validos.columns]
            df_treino = pd.concat([df_treino, df_validos[cols_comuns]], ignore_index=True)
            logs_incorporados = len(df_validos)

    X = df_treino[FEATURES].fillna(0)
    y = df_treino["churn_event"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    modelo_moat = GradientBoostingClassifier(
        n_estimators=180, learning_rate=0.06, max_depth=4, random_state=42
    )
    modelo_moat.fit(X_train, y_train)

    auc_score = roc_auc_score(y_test, modelo_moat.predict_proba(X_test)[:, 1])

    df_scored = df_base.copy()
    prob_churn = modelo_moat.predict_proba(df_scored[FEATURES].fillna(0))[:, 1]
    df_scored["moat_score"] = ((1 - prob_churn) * 100).round(1)

    def classificar_fortaleza(score):
        if score >= 75:
            return "Castelo Blindado"
        elif score >= 45:
            return "Em Consolidacao"
        return "Zona de Risco"

    def definir_next_best_action(row):
        s = row["moat_score"]
        if s < 45:
            if row["comodato_maquina"] == 1 and row["pull_through_ratio"] < 0.70:
                return "1. ALERTA MAQUINA: Auditar consumo minimo de insumos (pull-through < 70%)."
            if row["uso_credito_b2b"] < 0.30:
                return "1. BLINDAGEM FINANCEIRA: Liberar limite de credito pre-aprovado no App B2B."
            return "1. INTERVENCAO CS URGENTE: Agendar visita/reuniao de resgate e expandir SKUs."
        elif s < 75:
            if row["selo_cobranding"] == 0 and row["canal"] in ["Confeitaria", "Padaria", "Cafeteria"]:
                return "2. TRAVA DE MARCA: Oferecer Selo Oficial 'Feito com Nestle/Leite Moca'."
            if row["multi_threading_users"] < 2:
                return "2. MULTI-THREADING: Cadastrar +2 usuarios (Compras/Financeiro) no App."
            if row["comodato_maquina"] == 0:
                return "2. LOCK-IN FISICO: Propor comodato de maquina (Nescafe/Dispensador)."
            return "2. CROSS-SELL: Ofertar combo de +3 categorias complementares com desconto."
        else:
            if row["skus_distintos"] >= 9 or row["pull_through_ratio"] >= 1.15:
                return "3. UPSELL / TRUE-UP: Propor upgrade de faixa contratual + margem."
            return "3. RENOVACAO 24M: Blindar contrato plurianual com reajuste IPCA + 3%."

    df_scored["status_fortaleza"] = df_scored["moat_score"].apply(classificar_fortaleza)
    df_scored["next_best_action"] = df_scored.apply(definir_next_best_action, axis=1)

    df_importancia = (
        pd.DataFrame(
            {
                "Pilar / Feature": FEATURES,
                "Peso no Moat (%)": (modelo_moat.feature_importances_ * 100).round(2),
            }
        )
        .sort_values("Peso no Moat (%)", ascending=False)
        .reset_index(drop=True)
    )
    return auc_score, df_importancia, df_scored, logs_incorporados


@app.cell
def _(auc_score, canal_filtro, df_scored, logs_incorporados, mo, origem_dados):
    if canal_filtro.value != "Todos":
        df_view = df_scored[df_scored["canal"] == canal_filtro.value].copy()
    else:
        df_view = df_scored.copy()

    score_medio = df_view["moat_score"].mean()
    mrr_blindado = df_view[df_view["moat_score"] >= 75]["mrr_atual_brl"].sum()
    mrr_risco = df_view[df_view["moat_score"] < 45]["mrr_atual_brl"].sum()

    kpis = mo.hstack(
        [
            mo.stat(
                value=f"{score_medio:.1f} / 100",
                label="Moat Score Medio",
                caption=origem_dados,
            ),
            mo.stat(
                value=f"R$ {mrr_blindado:,.0f}",
                label="MRR Castelo Blindado (Score >= 75)",
                caption="Alta retencao e poder de reajuste",
            ),
            mo.stat(
                value=f"R$ {mrr_risco:,.0f}",
                label="MRR em Zona de Risco (Score < 45)",
                caption="Prioridade imediata para CS",
            ),
            mo.stat(
                value=f"{auc_score:.3f}",
                label="Precisao do Modelo (ROC-AUC)",
                caption=f"Logs Flywheel incorporados: {logs_incorporados}",
            ),
        ],
        widths="equal",
    )
    kpis
    return (df_view,)


@app.cell
def _(df_importancia, df_view, mo):
    df_acao = df_view.sort_values(by=["moat_score", "mrr_atual_brl"], ascending=[True, False])[
        [
            "pdv_id",
            "canal",
            "regiao",
            "mrr_atual_brl",
            "moat_score",
            "status_fortaleza",
            "next_best_action",
        ]
    ]

    csv_export = df_acao.to_csv(index=False).encode("utf-8-sig")
    btn_download = mo.download(
        data=csv_export,
        filename="lista_acao_cs_moat.csv",
        mimetype="text/csv",
        label="Baixar Lista de Acao CS (.csv)",
    )

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("### Peso de Cada Fortaleza no Modelo (Feature Importance)"),
                    mo.ui.table(df_importancia, selection=None),
                ]
            ),
            mo.vstack(
                [
                    mo.md("### Plano de Acao Prescritivo (Next Best Action)"),
                    btn_download,
                    mo.ui.table(df_acao.head(50), selection=None),
                ]
            ),
        ],
        widths=[1, 2],
        gap=2,
    )
    return


@app.cell
def _(df_view, mo):
    opcoes_pdv = df_view["pdv_id"].head(200).tolist()
    pdv_selecionado = mo.ui.dropdown(
        options=opcoes_pdv,
        value=opcoes_pdv[0] if opcoes_pdv else None,
        label="1. Selecione o PDV atendido:",
    )
    desfecho_acao = mo.ui.dropdown(
        options=["Sucesso", "Em Negociacao", "Sem Sucesso"],
        value="Sucesso",
        label="2. Resultado da intervencao de CS:",
    )
    btn_salvar_log = mo.ui.run_button(label="Gravar no Data Flywheel")

    mo.vstack(
        [
            mo.md(
                """
                ---
                ### Fechar o Data Flywheel (Registrar Resultado da Equipe)
                *Cada resultado gravado abaixo alimenta o arquivo local `flywheel_feedback_log.csv` para recalibrar o modelo quando voce clicar em **Retreinar Modelo** no topo.*
                """
            ),
            mo.hstack([pdv_selecionado, desfecho_acao, btn_salvar_log], justify="start", gap=2),
        ]
    )
    return btn_salvar_log, desfecho_acao, pdv_selecionado


@app.cell
def _(
    LOG_PATH,
    btn_salvar_log,
    datetime,
    desfecho_acao,
    df_scored,
    mo,
    os,
    pd,
    pdv_selecionado,
):
    mo.stop(not btn_salvar_log.value)

    linha_pdv = df_scored[df_scored["pdv_id"] == pdv_selecionado.value].iloc[0].to_dict()
    linha_pdv["desfecho"] = desfecho_acao.value
    linha_pdv["data_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_novo_log = pd.DataFrame([linha_pdv])
    if os.path.exists(LOG_PATH):
        df_existente = pd.read_csv(LOG_PATH)
        df_final = pd.concat([df_existente, df_novo_log], ignore_index=True)
    else:
        df_final = df_novo_log

    df_final.to_csv(LOG_PATH, index=False)

    mo.callout(
        mo.md(
            f"**Feedback registrado com sucesso!** PDV `{pdv_selecionado.value}` marcado como **{desfecho_acao.value}**. "
            f"Total de registros no Data Flywheel: **{len(df_final)}**. Clique em *Retreinar Modelo* no topo para atualizar os pesos."
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()

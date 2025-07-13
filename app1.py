import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

# --- 1. Configuração da Página e Paleta de Cores ---
st.set_page_config(layout="wide", page_title="Consultor de Portfólio Global")

# Paleta de cores baseada no logotipo (usada no histograma)
PALETA_LOGO = ['#3A4A6A', '#8FA4C5', '#5E7BAA', '#BCCBE2', '#6b7b93', '#2c3e50', '#d1d9e6']

# --- 2. Funções de Lógica e Cálculo ---
def get_allocation_by_profile(risk_profile):
    allocations_data = {
        "Conservador": {
            'Renda Fixa (Caixa/Curto Prazo EUA)': 40, 'Renda Fixa (Agregado EUA)': 35, 'Renda Fixa (Mercados Emergentes)': 5, 'Ações Americanas (Large Cap)': 10,
            'Ações Internacionais (Desenvolvidas)': 5, 'Ações Internacionais (Emergentes)': 0, 'Alternativos (Imobiliário/REITs)': 3,
            'Alternativos (Setor Específico/Commodities)': 2, 'Ações Americanas (Value/Qual/Growth)': 0
        }, "Moderado": {
            'Renda Fixa (Caixa/Curto Prazo EUA)': 15, 'Renda Fixa (Agregado EUA)': 15, 'Renda Fixa (Mercados Emergentes)': 10, 'Ações Americanas (Large Cap)': 32,
            'Ações Americanas (Value/Qual/Growth)': 10, 'Ações Internacionais (Desenvolvidas)': 7, 'Ações Internacionais (Emergentes)': 3,
            'Alternativos (Imobiliário/REITs)': 3, 'Alternativos (Setor Específico/Commodities)': 5
        }, "Arrojado": {
            'Renda Fixa (Caixa/Curto Prazo EUA)': 5, 'Renda Fixa (Agregado EUA)': 10, 'Renda Fixa (Mercados Emergentes)': 10, 'Ações Americanas (Large Cap)': 25,
            'Ações Americanas (Value/Qual/Growth)': 15, 'Ações Internacionais (Desenvolvidas)': 10, 'Ações Internacionais (Emergentes)': 10,
            'Alternativos (Imobiliário/REITs)': 5, 'Alternativos (Setor Específico/Commodities)': 10
        }, "Agressivo": {
            'Renda Fixa (Caixa/Curto Prazo EUA)': 0, 'Renda Fixa (Agregado EUA)': 5, 'Renda Fixa (Mercados Emergentes)': 10, 'Ações Americanas (Large Cap)': 20,
            'Ações Americanas (Value/Qual/Growth)': 20, 'Ações Internacionais (Desenvolvidas)': 15, 'Ações Internacionais (Emergentes)': 15,
            'Alternativos (Imobiliário/REITs)': 5, 'Alternativos (Setor Específico/Commodities)': 10
        }
    }
    all_keys = set(k for d in allocations_data.values() for k in d.keys())
    for profile_data in allocations_data.values():
        for key in all_keys:
            profile_data.setdefault(key, 0)
    return allocations_data.get(risk_profile, {})

def adjust_for_macro_scenario(base_allocation, scenario):
    adjustments = {}
    if scenario == "Juros Altos com Corte Eminente": adjustments = {'Renda Fixa (Caixa/Curto Prazo EUA)': -10, 'Renda Fixa (Agregado EUA)': 5, 'Ações Americanas (Large Cap)': 5}
    elif scenario == "Juros Altos sem Previsão de Corte": adjustments = {'Renda Fixa (Caixa/Curto Prazo EUA)': 15, 'Renda Fixa (Agregado EUA)': -5, 'Ações Americanas (Large Cap)': -5, 'Ações Americanas (Tech/Growth)': -5}
    elif scenario == "Juros Baixos sem Previsão de Subida": adjustments = {'Renda Fixa (Caixa/Curto Prazo EUA)': -10, 'Ações Americanas (Tech/Growth)': 5, 'Ações Internacionais (Emergentes)': 5}
    elif scenario == "Juros Baixos com Previsão de Subida": adjustments = {'Renda Fixa (Caixa/Curto Prazo EUA)': 10, 'Renda Fixa (Agregado EUA)': -10, 'Alternativos (Imobiliário/REITs)':-5}
    adjusted_allocation = base_allocation.copy()
    for asset, change in adjustments.items():
        if asset in adjusted_allocation: adjusted_allocation[asset] += change
    for asset in adjusted_allocation:
        if adjusted_allocation[asset] < 0: adjusted_allocation[asset] = 0
    total = sum(adjusted_allocation.values())
    if total > 0:
        for asset in adjusted_allocation: adjusted_allocation[asset] = (adjusted_allocation[asset] / total) * 100
    return adjusted_allocation

def plot_donut_chart(data_dict, title):
    if not data_dict: return None
    labels, sizes = list(data_dict.keys()), list(data_dict.values())
    
    # <<< ALTERADO: Cores revertidas para a paleta original do seu primeiro script
    colors = ['#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226', '#6a040f']
    
    colors = (colors * (len(sizes) // len(colors) + 1))[:len(sizes)]
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(aspect="equal"))
    wedges, _, autotexts = ax.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90, explode=[0.02]*len(sizes), pctdistance=0.85, wedgeprops=dict(width=0.4, edgecolor='w'))
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.legend(wedges, labels, title="Classes de Ativos", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12, title_fontsize=14)
    ax.set_title(title, size=18, weight='bold', pad=20)
    fig.tight_layout()
    return fig

def show_risk_questionnaire():
    st.header("1. Questionário de Perfil de Risco (Suitability)")
    with st.form("questionnaire_form"):
        score_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        questions = {
            "q1": ("Objetivo do Investimento", "Qual é o principal objetivo para esta carteira?", ["A) Preservar meu capital", "B) Gerar renda com crescimento moderado", "C) Obter um crescimento equilibrado", "D) Maximizar o crescimento do patrimônio"]),
            "q2": ("Tolerância a Flutuações", "Se sua carteira de R$1M perdesse 20%, qual sua reação?", ["A) Resgataria tudo", "B) Realocaria para algo mais seguro", "C) Manteria a estratégia", "D) Investiria mais"]),
            "q3": ("Horizonte de Tempo", "Quando precisará de parte significativa (25%+) deste capital?", ["A) Em menos de 2 anos", "B) Entre 2 e 5 anos", "C) Entre 6 e 10 anos", "D) Mais de 10 anos ou nunca"]),
            "q4": ("Experiência", "Como classifica seu conhecimento sobre investimentos?", ["A) Iniciante", "B) Tenho algum conhecimento", "C) Sou experiente", "D) Avançado/Especialista"]),
        }
        responses = {}
        for key, (subheader, question, options) in questions.items():
            st.subheader(subheader); responses[key] = st.radio(question, options, key=key)
        if st.form_submit_button("Calcular meu Perfil"):
            total_score = sum(score_mapping[resp.split(')')[0]] for resp in responses.values())
            if 5 <= total_score <= 7: st.session_state['risk_profile'] = "Conservador"
            elif 8 <= total_score <= 11: st.session_state['risk_profile'] = "Moderado"
            elif 12 <= total_score <= 15: st.session_state['risk_profile'] = "Arrojado"
            else: st.session_state['risk_profile'] = "Agressivo"
            st.session_state['score'] = total_score
            st.rerun()

@st.cache_data
def run_simplified_monte_carlo(initial_contribution, sip, years, mean_return, volatility, num_simulations=1000):
    num_months = years * 12
    monthly_mean = (1 + mean_return)**(1/12) - 1
    monthly_std = volatility / np.sqrt(12)
    portfolio_paths = np.zeros((num_months + 1, num_simulations))
    portfolio_paths[0, :] = initial_contribution
    for i in range(num_simulations):
        for m in range(1, num_months + 1):
            random_return = np.random.normal(monthly_mean, monthly_std)
            portfolio_paths[m, i] = (portfolio_paths[m-1, i] * (1 + random_return)) + sip
    return portfolio_paths

# --- 3. Interface Principal do Streamlit ---

# Cabeçalho com logotipo e título
logo_image = Image.open("LOGOTIPO_CARLOS ARNT RAMOS_5.png")
col1, col2 = st.columns([1, 4])
with col1:
    st.image(logo_image, width=250)
with col2:
    st.title("Consultor de Portfólio Global")
    st.markdown("Bem-vindo! Esta ferramenta combina seu **perfil de risco** com o **cenário macroeconômico** para sugerir um portfólio global.")

st.markdown("---")

if 'risk_profile' not in st.session_state:
    show_risk_questionnaire()
else:
    profile = st.session_state['risk_profile']
    score = st.session_state['score']
    st.success(f"### Seu Perfil de Risco: **{profile}** (Pontuação: {score})")
    if st.button("Refazer Questionário"):
        del st.session_state['risk_profile']; del st.session_state['score']; st.rerun()
    st.markdown("---")
    
    st.header(" 2. Cenário Macroeconômico (Tático)")
    macro_scenario = st.selectbox("Selecione o cenário de juros atual:", ["Juros Altos com Corte Eminente", 
                                                                          "Juros Altos sem Previsão de Corte", 
                                                                          "Juros Baixos sem Previsão de Subida", 
                                                                          "Juros Baixos com Previsão de Subida"])
    st.markdown("---")

    base_allocation = get_allocation_by_profile(profile)
    final_allocation = adjust_for_macro_scenario(base_allocation, macro_scenario)
    
    st.header(f" 3. Recomendação de Carteira Dinâmica")
    filtered_recs = {k: v for k, v in final_allocation.items() if v > 0.1}
    title = f"Alocação Tática - Perfil {profile}\nCenário: {macro_scenario.replace('','')}"
    st.pyplot(plot_donut_chart(filtered_recs, title), use_container_width=True)
    st.markdown("---")

    st.header(" 4. Simulação de Monte Carlo (Nominal vs. Real)")
    with st.form("simplified_simulation_form"):
        st.write("Defina os parâmetros do seu investimento para simular os resultados futuros.")
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_initial = st.number_input("Aporte Inicial (R$)", 0.0, value=50000.0, step=1000.0)
            sim_sip = st.number_input("Aportes Mensais (R$)", 0.0, value=1000.0, step=100.0)
        with col2:
            sim_years = st.number_input("Prazo em Anos", 1, 50, 20, 1)
            sim_mean_return = st.slider("Retorno Anual Esperado (%)", 0.0, 25.0, 12.0, 0.5) / 100
        with col3:
            sim_volatility = st.slider("Volatilidade Anual Esperada (%)", 0.0, 40.0, 18.0, 0.5) / 100
            sim_inflation = st.slider("Inflação Anual Esperada (%)", 0.0, 20.0, 4.5, 0.5) / 100
        submitted_sim = st.form_submit_button("🚀 Rodar Simulação")

    if 'simulation_results' not in st.session_state:
        st.session_state['simulation_results'] = {}

    if submitted_sim:
        with st.spinner("Rodando 1,000 simulações..."):
            nominal_paths = run_simplified_monte_carlo(sim_initial, sim_sip, sim_years, sim_mean_return, sim_volatility)
            monthly_inflation = (1 + sim_inflation)**(1/12) - 1
            months = np.arange(sim_years * 12 + 1)
            inflation_adjustment = (1 + monthly_inflation) ** months
            real_paths = nominal_paths / inflation_adjustment[:, np.newaxis]
            
            st.session_state['simulation_results'] = {
                'nominal_balances': nominal_paths[-1, :],
                'real_balances': real_paths[-1, :],
                'ran': True
            }

    if st.session_state['simulation_results'].get('ran'):
        results = st.session_state['simulation_results']
        final_balances_nominal = results['nominal_balances']
        final_balances_real = results['real_balances']

        results_data = {
            'Métrica': ['Portfolio End Balance (nominal)', 'Portfolio End Balance (real)'],
            '10th Percentile': [f"R$ {np.percentile(final_balances_nominal, 10):,.2f}", f"R$ {np.percentile(final_balances_real, 10):,.2f}"],
            '25th Percentile': [f"R$ {np.percentile(final_balances_nominal, 25):,.2f}", f"R$ {np.percentile(final_balances_real, 25):,.2f}"],
            '50th Percentile': [f"R$ {np.percentile(final_balances_nominal, 50):,.2f}", f"R$ {np.percentile(final_balances_real, 50):,.2f}"],
            '75th Percentile': [f"R$ {np.percentile(final_balances_nominal, 75):,.2f}", f"R$ {np.percentile(final_balances_real, 75):,.2f}"],
            '90th Percentile': [f"R$ {np.percentile(final_balances_nominal, 90):,.2f}", f"R$ {np.percentile(final_balances_real, 90):,.2f}"],
        }
        results_df = pd.DataFrame(results_data).set_index('Métrica')
        st.subheader("Resumo da Performance Simulada")
        st.table(results_df)

        st.subheader("Distribuição dos Resultados Finais")
        fig, ax = plt.subplots(figsize=(10, 6))

        combined_data = np.concatenate((final_balances_nominal, final_balances_real))
        
        median_value = np.percentile(combined_data, 50)
        ax.axvline(median_value, color='r', linestyle='--', linewidth=2, label=f'Mediana (P50): R$ {median_value:,.0f}')
        dist_to_lower = median_value - np.percentile(combined_data, 5)
        dist_to_upper = np.percentile(combined_data, 95) - median_value
        half_width = max(dist_to_lower, dist_to_upper)
        lower_bound = median_value - half_width
        upper_bound = median_value + half_width
        bins = np.linspace(lower_bound, upper_bound, 75)
        
        # Usando as cores do logo no histograma
        ax.hist(final_balances_nominal, bins=bins, label='Resultado Nominal', alpha=0.7, color=PALETA_LOGO[0]) # Azul escuro
        ax.hist(final_balances_real, bins=bins, label='Resultado Real (descontada a inflação)', alpha=0.7, color=PALETA_LOGO[1]) # Azul claro

        ax.set_xlim(lower_bound, upper_bound)
        
        def currency_formatter(x, pos):
            return f'R$ {x:,.0f}'
        ax.xaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        ax.set_title("Distribuição do Valor Final do Portfólio (Nominal vs. Real)")
        ax.set_xlabel("Valor do Patrimônio (R$)")
        ax.set_ylabel("Frequência (Número de Simulações)")
        
        ax.legend()
        plt.xticks(rotation=30, ha='right')
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("---")

        st.header(" 5. Projeção de Renda Passiva (Usufruto)")
        st.write("Com base no patrimônio simulado, veja quanto você poderia sacar anualmente.")
        
        with st.expander(" O que é a 'Regra dos 4%'? Clique para saber mais"):
            st.markdown("""
            A **"Regra dos 4%"** é resultado de um famoso estudo sobre finanças pessoais chamado "Trinity Study".

            #### O que o estudo fez?
            Pesquisadores analisaram dados históricos do mercado de ações e títulos dos EUA desde a década de 1920. Eles simularam qual seria a maior "taxa de saque" que uma pessoa poderia retirar de seu patrimônio no primeiro ano de aposentadoria (e depois corrigir esse valor pela inflação nos anos seguintes) sem esgotar o dinheiro ao longo de um período de 30 anos.

            #### A Conclusão
            Eles descobriram que uma taxa de saque inicial de **4%** tinha uma altíssima probabilidade de sucesso (o dinheiro não acabava em mais de 95% dos cenários históricos), desde que a carteira tivesse uma alocação significativa em ações (pelo menos 50%).

            Portanto, os 4% são um **ponto de partida empírico**, uma diretriz, e não uma garantia. É considerado um valor "seguro" porque sobreviveu a alguns dos piores períodos da história do mercado.
            """)
        
        swr_rate = st.slider("Selecione a Taxa de Saque Anual (%) - (Ex: Regra dos 4%)", 2.0, 7.0, 4.0, 0.1) / 100
        
        p10, p25, p50, p75, p90 = [np.percentile(final_balances_nominal, p) for p in [10, 25, 50, 75, 90]]

        withdrawal_data = {
            'Cenário (Percentil)': ['Pessimista (10º)', 'Abaixo da Média (25º)', 'Mediano (50º)', 'Acima da Média (75º)', 'Otimista (90º)'],
            'Patrimônio Nominal Final': [f"R$ {p:,.2f}" for p in [p10, p25, p50, p75, p90]],
            'Renda Anual Estimada': [f"R$ {(p * swr_rate):,.2f}" for p in [p10, p25, p50, p75, p90]],
            'Renda Mensal Estimada': [f"R$ {(p * swr_rate / 12):,.2f}" for p in [p10, p25, p50, p75, p90]]
        }
        withdrawal_df = pd.DataFrame(withdrawal_data).set_index('Cenário (Percentil)')

        st.table(withdrawal_df)
        st.caption(f"A 'Regra dos {swr_rate:.1%}' é um ponto de partida para estimar uma taxa de saque sustentável. A regra original sugere que, com uma taxa de 4%, há uma alta probabilidade do patrimônio durar por pelo menos 30 anos. Ajustes podem ser necessários dependendo do seu horizonte de tempo e tolerância a risco.")

st.markdown("---")
st.caption("Aviso: Esta é uma ferramenta educacional. As simulações são baseadas nos parâmetros fornecidos e não são garantia de resultados futuros.")

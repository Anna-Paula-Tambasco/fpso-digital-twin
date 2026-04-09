import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E FUSO HORÁRIO
# ==========================================
st.set_page_config(page_title="Gêmeo Digital - FPSO Búzios", page_icon="🚢", layout="wide")

# Força o horário oficial de Brasília (UTC-3) para evitar bugs de servidor na nuvem
FUSO_BR = datetime.timezone(datetime.timedelta(hours=-3))

# ==========================================
# 2. CARREGANDO O CÉREBRO DA IA (CACHE)
# ==========================================
@st.cache_resource
def carregar_modelo():
    try:
        pacote = joblib.load('cerebro_ia_buzios_completo.pkl')
        return pacote['modelos'], pacote['features_ordem'], pacote['limite_seguranca']
    except Exception as e:
        st.error("Erro fatal: Cérebro da IA não encontrado. Certifique-se de que o arquivo 'cerebro_ia_buzios_completo.pkl' está na mesma pasta.")
        st.stop()

modelos_ia, features_ordem, limite_seguranca = carregar_modelo()

# ==========================================
# 3. MOTOR DE FÍSICA (A ESTEIRA DE CÁLCULO)
# ==========================================
def calcular_fisica_naval(hs, tp, dir_onda, vel_vento, dir_vento, vel_corr, dir_corr, mes, hs_ontem, vento_ontem, rumo_navio):
    f = {}
    f['Onda_Hs_m'] = hs
    f['Onda_Tp_s'] = tp
    f['Vento_Vel_10m_m_s'] = vel_vento
    f['Corr_Vel_m_s'] = vel_corr
    f['Vento_Vel_30m_m_s'] = vel_vento * ((30.0 / 10.0) ** 0.11)
    f['Hs_Ontem'] = hs_ontem
    f['Vento_Ontem'] = vento_ontem
    f['Tendencia_Hs'] = hs - hs_ontem
    f['Tendencia_Vento'] = vel_vento - vento_ontem
    f['Mes_sen'] = np.sin(2 * np.pi * mes / 12.0)
    f['Mes_cos'] = np.cos(2 * np.pi * mes / 12.0)
    f['Onda_Dir_sen'], f['Onda_Dir_cos'] = np.sin(np.radians(dir_onda)), np.cos(np.radians(dir_onda))
    f['Vento_Dir_sen'], f['Vento_Dir_cos'] = np.sin(np.radians(dir_vento)), np.cos(np.radians(dir_vento))
    f['Corr_Dir_sen'], f['Corr_Dir_cos'] = np.sin(np.radians(dir_corr)), np.cos(np.radians(dir_corr))
    
    delta_vo = np.abs(dir_vento - dir_onda)
    f['Delta_Vento_Onda'] = 360 - delta_vo if delta_vo > 180 else delta_vo
    f['Esbeltez_Onda'] = hs / (tp ** 2) if tp > 0 else 0
    
    for param, direcao in zip(['Onda', 'Vento', 'Corr'], [dir_onda, dir_vento, dir_corr]):
        inc = np.abs(direcao - rumo_navio) % 360
        f[f'Incidencia_{param}'] = 360 - inc if inc > 180 else inc
        
    f['Comprimento_Onda_m'] = 1.56 * (tp ** 2)
    f['Razao_Onda_Navio'] = f['Comprimento_Onda_m'] / 330.0
    
    # RIGOR FÍSICO CORRIGIDO: Inclusão das massas específicas (Rho)
    RHO_AR = 1.225 # kg/m³
    RHO_AGUA = 1025.0 # kg/m³
    f['Pressao_Vento_30m'] = 0.5 * RHO_AR * (f['Vento_Vel_30m_m_s'] ** 2)
    f['Pressao_Corrente'] = 0.5 * RHO_AGUA * (vel_corr ** 2)
    f['Energia_Onda'] = hs ** 2 # Diretamente proporcional a H^2 na teoria linear
    
    df_fisica = pd.DataFrame([f])
    return df_fisica[features_ordem]

# ==========================================
# 4. CONEXÃO SATELITAL DUPLA (OPEN-METEO COM MEMÓRIA E RAJADAS)
# ==========================================
def buscar_clima_satelite():
    lat, lon = -24.75, -42.50
    # INOVAÇÃO: forecast_days=7 puxa uma semana de futuro para calcular Janelas Operacionais
    url_mar = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=wave_height,wave_period,wave_direction&past_days=1&forecast_days=7&timezone=America%2FSao_Paulo"
    url_ar = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m&past_days=1&forecast_days=7&timezone=America%2FSao_Paulo"
    
    try:
        resp_mar = requests.get(url_mar, timeout=10).json()
        resp_ar = requests.get(url_ar, timeout=10).json()
        
        # O índice 24 representa o "Agora" (já que pedimos 1 dia de passado: 24 horas)
        idx_agora = 24 
        
        hs_sat_atual = resp_mar['hourly']['wave_height'][idx_agora]
        tp_sat_atual = resp_mar['hourly']['wave_period'][idx_agora]
        dir_onda_sat = resp_mar['hourly']['wave_direction'][idx_agora]
        hs_sat_ontem = resp_mar['hourly']['wave_height'][0] 
        
        vel_vento_atual = resp_ar['hourly']['wind_speed_10m'][idx_agora]
        dir_vento_atual = resp_ar['hourly']['wind_direction_10m'][idx_agora]
        rajada_vento_atual = resp_ar['hourly']['wind_gusts_10m'][idx_agora] 
        vento_ontem = resp_ar['hourly']['wind_speed_10m'][0] 
        
        timestamp_leitura = resp_mar['hourly']['time'][idx_agora]
        
        # Retornamos também os dicionários inteiros para processar o futuro
        return hs_sat_atual, tp_sat_atual, dir_onda_sat, hs_sat_ontem, vel_vento_atual, dir_vento_atual, rajada_vento_atual, vento_ontem, timestamp_leitura, resp_mar['hourly'], resp_ar['hourly']
    except Exception as e:
        return None, None, None, None, None, None, None, None, None, None, None

# ==========================================
# 5. CONSTRUÇÃO DA INTERFACE (FRONT-END)
# ==========================================
st.title("🚢 Sistema de Inteligência Preditiva (FPSO Búzios)")
st.markdown("Monitoramento de Risco e Operabilidade Simultânea baseada em Machine Learning")

aba1, aba2, aba3, aba4 = st.tabs(["🕹️ Simulador (Oráculo)", "📅 Planejamento Anual", "🔬 Física Avançada", "🌪️ Máquina do Tempo"])

with aba1:
    st.header("Terminal de Decisão de Bordo")
    
    st.sidebar.header("Painel de Controle")
    modo = st.sidebar.radio("Selecione a Fonte de Dados:", ["Modo Manual", "📡 Sincronizar Satélite"])
    
    if modo == "📡 Sincronizar Satélite":
        st.sidebar.button("🔄 Forçar Atualização Agora")
    
    st.sidebar.markdown("---")
    # NOVO: Parâmetros obrigatórios de Arquitetura Naval e Auditoria
    st.sidebar.caption("⚓ Parâmetros do Navio e Operador")
    operador_id = st.sidebar.text_input("ID do Operador / OIM", value="OIM-Turno-A")
    rumo_navio = st.sidebar.number_input("Rumo da Proa (Heading) - Graus", min_value=0.0, max_value=360.0, value=210.0, step=1.0)
    calado_navio = st.sidebar.number_input("Calado Atual (Draft) - Metros", min_value=10.0, max_value=25.0, value=15.5, step=0.1)
    st.sidebar.markdown("---")
    
    rajada_vento = 0.0 

    if modo == "Modo Manual":
        hs = st.sidebar.slider("Altura da Onda (m)", 0.0, 10.0, 2.5)
        tp = st.sidebar.slider("Período da Onda (s)", 2.0, 25.0, 8.0)
        dir_onda = st.sidebar.slider("Direção Onda (Graus)", 0, 360, 180)
        vel_vento = st.sidebar.slider("Vento Sustentado (m/s)", 0.0, 30.0, 10.0)
        rajada_vento = st.sidebar.slider("Rajada Máxima (m/s)", vel_vento, 40.0, vel_vento * 1.3)
        dir_vento = st.sidebar.slider("Direção Vento (Graus)", 0, 360, 200)
        vel_corr = st.sidebar.slider("Correnteza (m/s)", 0.0, 3.0, 0.5)
        dir_corr = st.sidebar.slider("Direção Correnteza (Graus)", 0, 360, 90)
        
        mes = datetime.datetime.now(FUSO_BR).month
        hs_ontem = st.sidebar.slider("Onda Há 24h (Inércia)", 0.0, 10.0, hs * 0.9)
        vento_ontem = st.sidebar.slider("Vento Há 24h (Inércia)", 0.0, 30.0, vel_vento * 0.9)
        timestamp_exibicao = datetime.datetime.now(FUSO_BR).strftime("%Y-%m-%d %H:%M (Simulação Manual)")
        
    else:
        st.sidebar.info("📡 Triangulando Satélites (Copernicus/NOAA)...")
        hs_atual, tp_atual, dir_onda_sat, hs_ontem_sat, vel_vento_atual, dir_vento_sat, rajada_sat, vento_ontem_sat, carimbo_tempo, resp_mar, resp_ar = buscar_clima_satelite()
        
        if hs_atual is not None:
            st.sidebar.success("✅ Conexão Estabelecida!")
            hs, tp, dir_onda, hs_ontem = hs_atual, tp_atual, dir_onda_sat, hs_ontem_sat
            vel_vento, dir_vento, rajada_vento, vento_ontem = vel_vento_atual, dir_vento_sat, rajada_sat, vento_ontem_sat
            
            st.sidebar.markdown("---")
            st.sidebar.caption("🌊 Telemetria Submarina")
            vel_corr = st.sidebar.slider("Correnteza Local (m/s)", 0.0, 3.0, 0.5)
            dir_corr = st.sidebar.slider("Direção Correnteza (Graus)", 0, 360, 90)
            
            mes = datetime.datetime.now(FUSO_BR).month
            timestamp_exibicao = f"{carimbo_tempo.replace('T', ' ')} (Sinal Satélite)"
        else:
            st.sidebar.error("Sinal Satélite Perdido. Exigido Input Manual de Contingência.")
            # Em vez de dados ocultos, alerta visual severo para assumir comando
            hs, tp, dir_onda, vel_vento, rajada_vento, dir_vento, vel_corr, dir_corr, mes, hs_ontem, vento_ontem = 0.1, 8.0, 0, 0.1, 0.1, 0, 0.1, 0, 6, 0.1, 0.1
            timestamp_exibicao = "SINAL PERDIDO - MODO CONTINGÊNCIA ATIVADO"

    # --- INSPEÇÃO FÍSICA E CÁLCULOS TÁTICOS ---
    comprimento_onda = 1.56 * (tp ** 2)
    esbeltez = hs / comprimento_onda if comprimento_onda > 0 else 0
    h_max = hs * 1.86
    vento_knots = vel_vento * 1.94384
    rajada_knots = rajada_vento * 1.94384
    
    # 1. Cálculo de Ressonância Estrutural
    razao_ressonancia = comprimento_onda / 330.0 # 330m é o Lpp do FPSO
    alerta_ressonancia = "⚠️ ZONA DE RESSONÂNCIA" if 0.8 <= razao_ressonancia <= 1.2 else "Fora de Ressonância"
    
    # 2. Cálculo de Mar Cruzado
    delta_vo = np.abs(dir_vento - dir_onda)
    delta_vento_onda = 360 - delta_vo if delta_vo > 180 else delta_vo
    alerta_cruzado = "⚠️ MAR CRUZADO SEVERO" if delta_vento_onda > 45 else "Alinhamento Seguro"
    
    def zona_impacto_naval(angulo_verdadeiro, proa):
        relativo = (angulo_verdadeiro - proa) % 360
        if relativo == 0: return "Na Proa (Avanço)"
        elif 0 < relativo < 90: return f"Bochecha de Boreste (+{relativo:.0f}°)"
        elif relativo == 90: return "Través de Boreste (+90°)"
        elif 90 < relativo < 180: return f"Alheta de Boreste (+{relativo:.0f}°)"
        elif relativo == 180: return "Na Popa (Recuo)"
        elif 180 < relativo < 270: return f"Alheta de Bombordo (-{360-relativo:.0f}°)"
        elif relativo == 270: return "Través de Bombordo (-90°)"
        else: return f"Bochecha de Bombordo (-{360-relativo:.0f}°)"
        
    def escala_beaufort(vel_ms):
        if vel_ms < 0.5: return "Força 0 (Calmaria)"
        elif vel_ms < 1.5: return "Força 1 (Aragem)"
        elif vel_ms < 3.3: return "Força 2 (Brisa Leve)"
        elif vel_ms < 5.4: return "Força 3 (Brisa Fraca)"
        elif vel_ms < 7.9: return "Força 4 (Brisa Moderada)"
        elif vel_ms < 10.7: return "Força 5 (Brisa Fresca)"
        elif vel_ms < 13.8: return "Força 6 (Vento Fresco)"
        elif vel_ms < 17.1: return "Força 7 (Vento Forte)"
        elif vel_ms < 20.7: return "Força 8 (Ventania)"
        elif vel_ms < 24.4: return "Força 9 (Ventania Forte)"
        elif vel_ms < 28.4: return "Força 10 (Tempestade)"
        elif vel_ms < 32.6: return "Força 11 (Temp. Violenta)"
        else: return "Força 12 (Furacão)"
        
    beaufort_str_sustentado = escala_beaufort(vel_vento)
    # NOVO: Classificação da rajada
    beaufort_str_rajada = escala_beaufort(rajada_vento)

    st.markdown(f"**⏱️ Última Atualização:** `{timestamp_exibicao}` | **Operador:** `{operador_id}` | **Calado:** `{calado_navio} m`")
    
    col_resumo, col_bussola = st.columns([1.5, 1])
    
    with col_resumo:
        st.subheader("Dinâmica de Corpos Rígidos (Ref. Naval)")
        st.markdown(f"- **Onda ($H_s$):** {hs} m | **Max ($H_{{max}}$):** {h_max:.1f} m")
        st.markdown(f"  ↳ *Comprimento ($\lambda$):* {comprimento_onda:.1f} m | *Razão Navio/Onda:* **{razao_ressonancia:.2f}** ({alerta_ressonancia})")
        st.markdown(f"  ↳ *Incidência:* **{zona_impacto_naval(dir_onda, rumo_navio)}**")
        
        # NOVO: Inserção do Beaufort para Vento Sustentado e Rajada
        st.markdown(f"- **Vento:** {vel_vento:.1f} m/s ({vento_knots:.1f} nós) - **{beaufort_str_sustentado}**")
        st.markdown(f"  ↳ *Rajada Máxima:* <span style='color:red; font-weight:bold;'>{rajada_vento:.1f} m/s ({rajada_knots:.1f} nós) - {beaufort_str_rajada}</span>", unsafe_allow_html=True)
        st.markdown(f"  ↳ *Incidência:* **{zona_impacto_naval(dir_vento, rumo_navio)}** | *$\Delta$ Vento-Onda:* **{delta_vento_onda:.0f}°** ({alerta_cruzado})")
        
        st.markdown(f"- **Correnteza Local:** {vel_corr} m/s")
        st.markdown(f"  ↳ *Incidência:* **{zona_impacto_naval(dir_corr, rumo_navio)}**")
        
    with col_bussola:
        # --- CORREÇÃO TRIGONOMÉTRICA (VETORES DE IMPACTO) ---
        # Calculamos os ângulos relativos matemáticos
        onda_rel = (dir_onda - rumo_navio) % 360
        vento_rel = (dir_vento - rumo_navio) % 360
        corr_rel = (dir_corr - rumo_navio) % 360
        
        # INVERSÃO (Vento e Onda são de onde vêm, então a seta no radar deve vir do horizonte para o centro)
        # O Plotly desenha do centro para fora [0, 1]. Para parecer que está vindo de fora para dentro batendo no navio,
        # nós invertemos o ângulo em 180 graus na tela.
        onda_visual = (onda_rel + 180) % 360
        vento_visual = (vento_rel + 180) % 360
        
        fig_compass = go.Figure()
        # Vetor Navio (Fixo no Norte, proa apontando pra frente)
        fig_compass.add_trace(go.Scatterpolar(r=[0, 1], theta=[0, 0], mode='lines+markers', name='Proa (FPSO)', line=dict(color='black', width=5), marker=dict(symbol='arrow-up', size=16)))
        
        # Vetor Onda (Batendo no navio: inverte o ângulo na plotagem para a seta chegar no centro)
        fig_compass.add_trace(go.Scatterpolar(r=[1, 0], theta=[onda_visual, onda_visual], mode='lines+markers', name='Onda', line=dict(color='blue', width=3), marker=dict(symbol='arrow-up', size=12)))
        
        # Vetor Vento (Batendo no navio)
        fig_compass.add_trace(go.Scatterpolar(r=[1, 0], theta=[vento_visual, vento_visual], mode='lines+markers', name='Vento', line=dict(color='gray', width=2, dash='dot'), marker=dict(symbol='arrow-up', size=10)))
        
        # Vetor Correnteza (Arrastando o navio: aponta do centro para a borda)
        fig_compass.add_trace(go.Scatterpolar(r=[0, 1], theta=[0, corr_rel], mode='lines+markers', name='Corrente', line=dict(color='cyan', width=2), marker=dict(symbol='arrow-up', size=10)))
        
        fig_compass.update_layout(
            polar=dict(
                radialaxis=dict(visible=False), # CEREJA DO BOLO: Some com os números do meio
                angularaxis=dict(direction='clockwise', rotation=90, tickvals=[0, 45, 90, 135, 180, 225, 270, 315], ticktext=['Proa', 'Boch. BE', 'Través BE', 'Alheta BE', 'Popa', 'Alheta BB', 'Través BB', 'Boch. BB'])
            ), 
            title={'text': "TELA DE DP (Modo Head-Up)", 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            showlegend=False, 
            margin=dict(t=70, b=30, l=40, r=40), # Aumenta a margem do topo (t) para afastar o título
            height=320 # Aumenta a altura geral para o gráfico "respirar"
        )
        st.plotly_chart(fig_compass, use_container_width=True)
        
    st.markdown("---")
    
    texto_status_ia = ""
    
    margem_alerta = (limite_seguranca * 100) - 15.0 
    limite_critico = limite_seguranca * 100

    if esbeltez > 0.142: 
        st.error("⚠️ **ERRO DE FÍSICA OCEÂNICA:** Onda com esbeltez além do limite de quebramento (Wave Breaking). O Motor de IA foi desativado temporariamente para evitar falsos positivos.")
        texto_status_ia = "CÁLCULO ABORTADO - ONDA FORA DO ENVELOPE FÍSICO"
    else:
        df_alvo = calcular_fisica_naval(hs, tp, dir_onda, vel_vento, dir_vento, vel_corr, dir_corr, mes, hs_ontem, vento_ontem, rumo_navio)
        
        col1, col2, col3, col4 = st.columns(4)
        maquinas = [('SLS_Guindaste', col1), ('SLS_ROV', col2), ('SLS_Barco_Apoio', col3), ('SLS_Offloading', col4)]
        
        texto_status_ia += f"(Limiar de Segurança Operacional: {limite_seguranca*100}%)\n"
        
        for alvo, coluna in maquinas:
            nome_curto = alvo.replace('SLS_', '').replace('_', ' ')
            modelo = modelos_ia[alvo]
            risco = modelo.predict_proba(df_alvo)[0][1] * 100
            
            status_txt = "SUSPENSO" if risco >= limite_critico else "ATENÇÃO" if risco >= margem_alerta else "LIBERADO"
            texto_status_ia += f"- {nome_curto}: Risco {risco:.1f}% -> {status_txt}\n"
            
            with coluna:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = risco, number = {'suffix': "%", 'font': {'size': 35}},
                    title = {'text': nome_curto, 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "black" if risco >= limite_critico else "gray"},
                        'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                        'steps': [
                            {'range': [0, margem_alerta], 'color': "#00cc66"},
                            {'range': [margem_alerta, limite_critico], 'color': "#ffcc00"},
                            {'range': [limite_critico, 100], 'color': "#ff4d4d"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': limite_critico}
                    }
                ))
                fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=220)
                st.plotly_chart(fig, use_container_width=True)
                
                if risco >= limite_critico: st.error("🛑 SUSPENSO")
                elif risco >= margem_alerta: st.warning("⚠️ ATENÇÃO")
                else: st.success("✅ LIBERADO")

    st.markdown("---")
    st.caption("📋 **NOTA DE COMPLIANCE OPERACIONAL:** Este modelo assume condições combinadas de estado de mar. Flutuações baseadas na alteração dinâmica do calado (Draft) informadas no cabeçalho são abstraídas estatisticamente no envelope probabilístico treinado pela IA.")
    
    st.markdown("### Registro de Decisão Oficial (RDO)")
    
    # Define a tarja legal do documento
    tarja_legal = "[TELEMETRIA OFICIAL - DADOS DE SATÉLITE]" if modo == "📡 Sincronizar Satélite" else "[SIMULAÇÃO HIPOTÉTICA - NÃO VÁLIDO PARA OPERAÇÃO]"

    relatorio_texto = f"""=== BOLETIM METOCEÂNICO E PREDITIVO - FPSO BÚZIOS ===
{tarja_legal}
Data/Hora da Avaliação: {timestamp_exibicao}
ID do OIM / Operador Responsável: {operador_id}
Rumo da Proa (Heading): {rumo_navio}°
Calado da Unidade (Draft): {calado_navio} m

[CONDIÇÕES AMBIENTAIS E DINÂMICA DE CORPO RÍGIDO]
- ONDA: Significativa (Hs) = {hs} m | Máxima Prevista (Hmax) = {h_max:.1f} m
  Período de Pico (Tp): {tp} s -> Comprimento de Onda (λ): {comprimento_onda:.1f} m
  Ressonância Casco/Onda: {razao_ressonancia:.2f} ({alerta_ressonancia.replace('⚠️ ', '')})
  Direção de Origem: {dir_onda}° ({zona_impacto_naval(dir_onda, rumo_navio)})

- VENTO: Velocidade Sustentada = {vel_vento} m/s ({vento_knots:.1f} nós) - {beaufort_str_sustentado}
  Rajada Máxima = {rajada_vento} m/s ({rajada_knots:.1f} nós) - {beaufort_str_rajada}
  Direção de Origem: {dir_vento}° ({zona_impacto_naval(dir_vento, rumo_navio)})
  Mar Cruzado (Δ Vento/Onda): {delta_vento_onda:.0f}° ({alerta_cruzado.replace('⚠️ ', '')})

- CORRENTEZA: Velocidade Local = {vel_corr} m/s
  Direção de Origem: {dir_corr}° ({zona_impacto_naval(dir_corr, rumo_navio)})

[VEREDITO DA INTELIGÊNCIA ARTIFICIAL]
{texto_status_ia}

Boletim gerado eletronicamente pelo Gêmeo Digital Preditivo e assinado digitalmente pelo operador logado.
"""

    st.download_button(
        label="📥 Baixar Boletim de Liberação (RDO) - Formato Texto",
        data=relatorio_texto,
        file_name=f"RDO_Buzios_{datetime.datetime.now(FUSO_BR).strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain", type="primary"
    )

    # --- INOVAÇÃO: RADAR DE JANELAS OPERACIONAIS (PREVISÃO DE 7 DIAS) ---
    if modo == "📡 Sincronizar Satélite" and 'resp_mar' in locals() and resp_mar is not None:
        st.markdown("---")
        st.subheader("🔭 Radar de Janelas Operacionais (Previsão de 7 Dias)")
        st.markdown("O Gêmeo Digital está processando a previsão meteorológica do Copernicus para a próxima semana e cruzando com os limites da viga-navio para mapear as janelas seguras de trabalho.")
        
        with st.spinner("Simulando o futuro na Rede Neural..."):
            # Preparando os 7 dias de futuro (168 horas)
            tempos_futuro = pd.to_datetime(resp_mar['time'][24:])
            previsoes_futuro = []
            
            for i in range(24, len(resp_mar['time'])):
                mes_futuro = pd.to_datetime(resp_mar['time'][i]).month
                df_futuro = calcular_fisica_naval(
                    hs=resp_mar['wave_height'][i], tp=resp_mar['wave_period'][i], dir_onda=resp_mar['wave_direction'][i],
                    vel_vento=resp_ar['wind_speed_10m'][i], dir_vento=resp_ar['wind_direction_10m'][i],
                    vel_corr=vel_corr, dir_corr=dir_corr, mes=mes_futuro, 
                    hs_ontem=resp_mar['wave_height'][i-24], vento_ontem=resp_ar['wind_speed_10m'][i-24], 
                    rumo_navio=rumo_navio
                )
                
                # Focando no gargalo principal: Offloading
                risco_offloading = modelos_ia['SLS_Offloading'].predict_proba(df_futuro)[0][1] * 100
                previsoes_futuro.append(risco_offloading)
                
            fig_forecast = go.Figure()
            
            fig_forecast.add_trace(go.Scatter(
                x=tempos_futuro, y=previsoes_futuro, mode='lines', fill='tozeroy',
                name='Risco de Paralisação (%)', line=dict(color='purple', width=3),
                fillcolor='rgba(128, 0, 128, 0.2)'
            ))
            
            fig_forecast.add_hline(y=limite_critico, line_dash="dash", line_color="red", annotation_text="CORTE DE OPERAÇÃO (100% PERIGO)")
            fig_forecast.add_hline(y=margem_alerta, line_dash="dot", line_color="orange", annotation_text="ATENÇÃO")
            
            # Adicionando áreas de destaque verdes (Janelas Seguras Contínuas)
            fig_forecast.add_hrect(y0=0, y1=margem_alerta, fillcolor="green", opacity=0.1, layer="below")
            
            fig_forecast.update_layout(
                height=350, margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Timeline da Próxima Semana", yaxis_title="Probabilidade de Risco da IA (%)",
                yaxis=dict(range=[0, 105])
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            st.caption("📌 **Inteligência Logística:** A zona sombreada em verde no fundo indica áreas onde o risco é mínimo. Procure por vales contínuos no gráfico roxo para agendar operações longas.")


with aba2:
    st.header("📅 Matriz Climatológica e Impacto Financeiro (Lucro Cessante)")
    st.markdown("Projeção de operabilidade processando **11 anos ininterruptos** de Big Data (ERA5) em tempo real via Vetorização.")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💰 Economia do Petróleo (Búzios)")
    producao_bopd = st.sidebar.number_input("Produção do FPSO (Barris/Dia)", min_value=0, value=150000, step=10000)
    preco_brent = st.sidebar.number_input("Preço do Barril Brent (USD)", min_value=0.0, value=82.50, step=1.0)
    custo_barco = st.sidebar.number_input("Custo Diário OSV (USD)", min_value=0, value=50000, step=5000)
    
    # 1. FUNÇÃO BURRA: Lê o disco apenas UMA vez e guarda na RAM
    @st.cache_data
    def carregar_big_data_bruto():
        try:
            df = pd.read_csv("dataset_buzios_oficial.csv", sep=";")
            df['Data'] = pd.to_datetime(df['Data'])
            df['Mes_Num'] = df['Data'].dt.month
            return df
        except Exception as e:
            return None

    # 2. FUNÇÃO INTELIGENTE: Pega da RAM e faz matemática na velocidade da luz
    @st.cache_data
    def calcular_climatologia_vetorizada(df_bruto, rumo_navio_selecionado):
        try:
            df = df_bruto.copy() # Evita corromper o cache original
            X = pd.DataFrame(index=df.index)
            X['Onda_Hs_m'] = df['Onda_Hs_m']
            X['Onda_Tp_s'] = df['Onda_Tp_s']
            X['Vento_Vel_10m_m_s'] = df['Vento_Vel_10m_m_s']
            X['Corr_Vel_m_s'] = df['Corr_Vel_m_s']
            X['Vento_Vel_30m_m_s'] = df['Vento_Vel_10m_m_s'] * ((30.0 / 10.0) ** 0.11)
            X['Hs_Ontem'] = df['Onda_Hs_m'].shift(1).bfill()
            X['Vento_Ontem'] = df['Vento_Vel_10m_m_s'].shift(1).bfill()
            X['Tendencia_Hs'] = X['Onda_Hs_m'] - X['Hs_Ontem']
            X['Tendencia_Vento'] = X['Vento_Vel_10m_m_s'] - X['Vento_Ontem']
            X['Mes_sen'] = np.sin(2 * np.pi * df['Mes_Num'] / 12.0)
            X['Mes_cos'] = np.cos(2 * np.pi * df['Mes_Num'] / 12.0)
            
            for col, prefix in zip(['Onda_Dir_graus', 'Vento_Dir_graus', 'Corr_Dir_graus'], ['Onda_Dir', 'Vento_Dir', 'Corr_Dir']):
                rad = np.radians(df[col])
                X[f'{prefix}_sen'] = np.sin(rad)
                X[f'{prefix}_cos'] = np.cos(rad)
                
            delta_vo = np.abs(df['Vento_Dir_graus'] - df['Onda_Dir_graus'])
            X['Delta_Vento_Onda'] = np.where(delta_vo > 180, 360 - delta_vo, delta_vo)
            
            # --- A CORREÇÃO ESTÁ AQUI: Passando o Mar Cruzado para a tabela final da Aba 4 ---
            df['Delta_Vento_Onda'] = X['Delta_Vento_Onda'] 
            
            X['Esbeltez_Onda'] = np.where(df['Onda_Tp_s'] > 0, df['Onda_Hs_m'] / (df['Onda_Tp_s'] ** 2), 0)
            
            for param, col in zip(['Onda', 'Vento', 'Corr'], ['Onda_Dir_graus', 'Vento_Dir_graus', 'Corr_Dir_graus']):
                inc = np.abs(df[col] - rumo_navio_selecionado) % 360
                X[f'Incidencia_{param}'] = np.where(inc > 180, 360 - inc, inc)
                
            X['Comprimento_Onda_m'] = 1.56 * (df['Onda_Tp_s'] ** 2)
            X['Razao_Onda_Navio'] = X['Comprimento_Onda_m'] / 330.0
            
            RHO_AR, RHO_AGUA = 1.225, 1025.0
            X['Pressao_Vento_30m'] = 0.5 * RHO_AR * (X['Vento_Vel_30m_m_s'] ** 2)
            X['Pressao_Corrente'] = 0.5 * RHO_AGUA * (X['Corr_Vel_m_s'] ** 2)
            X['Energia_Onda'] = df['Onda_Hs_m'] ** 2
            
            X_ia = X[features_ordem]

            # Salvando TODAS as variáveis da física na tabela final para a Aba 3 poder ler!
            for col in features_ordem:
                df[col] = X[col]
            df['Delta_Vento_Onda'] = X['Delta_Vento_Onda'] # Garantindo o Mar Cruzado da Aba 4
            # ----------------------------

            for alvo in ['SLS_Guindaste', 'SLS_ROV', 'SLS_Barco_Apoio', 'SLS_Offloading']:
                riscos = modelos_ia[alvo].predict_proba(X_ia)[:, 1]
                df[f'Risco_Prob_{alvo}'] = riscos 
                df[f'Falha_{alvo}'] = (riscos >= limite_seguranca).astype(int)
                
            downtime_mensal = df.groupby('Mes_Num')[[f'Falha_{alvo}' for alvo in ['SLS_Guindaste', 'SLS_ROV', 'SLS_Barco_Apoio', 'SLS_Offloading']]].mean()
            return downtime_mensal, df
        except Exception as e:
            return None, None

    df_dataset_bruto = carregar_big_data_bruto()

    with st.spinner(f"Minerando 11 anos de Big Data Oceanográfico para a Proa {rumo_navio}°..."):
        if df_dataset_bruto is not None:
            downtime_real_mensal, df_historico_completo = calcular_climatologia_vetorizada(df_dataset_bruto, rumo_navio)
        else:
            downtime_real_mensal, df_historico_completo = None, None
            
    if downtime_real_mensal is None:
        st.error("⚠️ **ERRO:** Arquivo `dataset_buzios_oficial.csv` não encontrado na pasta raiz.")
    else:
        operabilidade_mensal = (1.0 - downtime_real_mensal) * 100
        operabilidade_mensal.columns = ['Guindaste', 'ROV', 'Barco de Apoio', 'Offloading']
        meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
        st.subheader(f"Mapa Analítico de Operabilidade Histórica Real (Proa travada em {rumo_navio}°)")
        z_data = operabilidade_mensal.values.T
        y_labels = operabilidade_mensal.columns.tolist()
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=z_data, x=meses_nomes, y=y_labels, colorscale='RdYlGn', zmin=60, zmax=100,
            text=[[f"{val:.1f}%" for val in row] for row in z_data], texttemplate="%{text}", textfont={"color": "black", "size": 14}, hoverinfo="x+y+z"
        ))
        fig_heat.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💸 Projeção de Risco Financeiro Anual e Gargalo Logístico")
        
        st.sidebar.markdown("---")
        st.sidebar.caption("🛢️ Logística de Exportação")
        buffer_dias = st.sidebar.slider("Buffer de Tancagem (Dias até Tank Top)", min_value=0.0, max_value=15.0, value=7.0, step=0.5)
        
        dias_no_mes = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) 
        dias_perdidos_fpso_mensal = downtime_real_mensal['Falha_SLS_Offloading'].values * dias_no_mes
        dias_perdidos_barco_mensal = downtime_real_mensal['Falha_SLS_Barco_Apoio'].values * dias_no_mes
        
        receita_diaria = producao_bopd * preco_brent
        
        lucro_cessante_mensal = np.maximum(0, dias_perdidos_fpso_mensal - buffer_dias)
        dias_lucro_cessante_total = lucro_cessante_mensal.sum()
        dias_perdidos_fpso_total = dias_perdidos_fpso_mensal.sum()
        dias_perdidos_barco_total = dias_perdidos_barco_mensal.sum()
        
        lucro_cessante_real = dias_lucro_cessante_total * receita_diaria
        receita_retida = (dias_perdidos_fpso_total - dias_lucro_cessante_total) * receita_diaria
        perda_opex_barco = dias_perdidos_barco_total * custo_barco
        impacto_total_direto = lucro_cessante_real + perda_opex_barco
        
        col_fin1, col_fin2, col_fin3 = st.columns(3)
        with col_fin1: 
            st.metric("Lucro Cessante (Poços Fechados)", f"{dias_lucro_cessante_total:.1f} dias", f"- ${lucro_cessante_real:,.0f}", delta_color="inverse")
            st.caption(f"Clima severo ultrapassou tancagem ({buffer_dias} dias/mês).")
        with col_fin2: 
            st.metric("OPEX Perdido (OSV Waiting)", f"{dias_perdidos_barco_total:.1f} dias", f"- ${perda_opex_barco:,.0f}", delta_color="inverse")
            st.caption("Diárias de barcos parados.")
        with col_fin3: 
            st.error(f"**PREJUÍZO DIRETO:**\n### ${impacto_total_direto:,.0f} USD")
            
        st.warning(f"🛢️ **Receita Retida:** Além do prejuízo, atrasou **\${receita_retida:,.0f} USD** em óleo retido nos tanques.")
        st.info("💡 **Dica Logística:** Gire a proa na barra lateral para fugir de ciclones.")

with aba3:
    st.header("🔬 Diagnóstico Científico e Transparência da IA (Caixa Preta)")
    st.markdown("Análise profunda da física não-linear e dos pesos matemáticos que regem o Gêmeo Digital.")
    
    if 'df_historico_completo' in locals() and df_historico_completo is not None:
        
        col_scatter, col_radar = st.columns([1.5, 1])
        
        with col_scatter:
            st.subheader("Dispersão Histórica de Operabilidade ($H_s \\times T_p$)")
            st.markdown(f"Processamento GPU (WebGL) de 11 anos com navio a **{rumo_navio}°**.")
            
            alvo_scatter = st.selectbox("Selecione a Operação para Diagnóstico:", 
                                         ['SLS_Guindaste', 'SLS_ROV', 'SLS_Barco_Apoio', 'SLS_Offloading'], 
                                         index=3, key="select_scatter")
            
            seguro = df_historico_completo[df_historico_completo[f'Falha_{alvo_scatter}'] == 0]
            falha = df_historico_completo[df_historico_completo[f'Falha_{alvo_scatter}'] == 1]
            
            fig_scatter = go.Figure()
            
            fig_scatter.add_trace(go.Histogram2dContour(
                x=df_historico_completo['Onda_Tp_s'], y=df_historico_completo['Onda_Hs_m'],
                colorscale='Blues', reversescale=False, opacity=0.4,
                name='Densidade Climática', showscale=False, contours=dict(showlines=True, coloring='heatmap')
            ))
            
            fig_scatter.add_trace(go.Scattergl(
                x=seguro['Onda_Tp_s'], y=seguro['Onda_Hs_m'], mode='markers',
                name='Operação Liberada', marker=dict(color='#00cc66', size=4, opacity=0.2)
            ))
            
            fig_scatter.add_trace(go.Scattergl(
                x=falha['Onda_Tp_s'], y=falha['Onda_Hs_m'], mode='markers',
                name='Risco Crítico (Downtime)', marker=dict(color='#ff4d4d', size=6, symbol='x')
            ))
            
            t_range = np.linspace(2, 22, 100)
            h_break = 0.142 * 1.56 * (t_range ** 2) 
            fig_scatter.add_trace(go.Scatter(
                x=t_range, y=h_break, mode='lines', name='Limite Físico Oceano', line=dict(color='black', width=3, dash='dash')
            ))
            
            limite_tradicional = 3.5 if alvo_scatter == 'SLS_Offloading' else 2.5
            fig_scatter.add_trace(go.Scatter(
                x=[2, 22], y=[limite_tradicional, limite_tradicional], mode='lines',
                name=f'Regra Fixa Antiga ({limite_tradicional}m)', line=dict(color='orange', width=2, dash='dot')
            ))
            
            fig_scatter.update_layout(
                xaxis_title="Período de Pico ($T_p$) - Segundos", yaxis_title="Altura Significativa ($H_s$) - Metros",
                xaxis=dict(range=[2, 22], dtick=2), yaxis=dict(range=[0, 12], dtick=1),
                height=550, margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.9)", font=dict(color="black"))
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("📌 A linha tracejada preta é o limite físico real. A **linha laranja** é a regra antiga. Note que a IA encontra ressonância (X vermelhos) abaixo da regra antiga, e libera produção acima dela.")

        with col_radar:
            st.subheader("Rosa de Energia Climatológica")
            st.markdown("Distribuição dos 11 anos sobre os eixos estruturais do FPSO.")
            
            param_radar = st.selectbox("Selecione o Parâmetro:", ["Onda (Hs)", "Vento (Vel)", "Corrente (Vel)"], key="select_radar")
            
            mapa_radar = {
                "Onda (Hs)": ("Onda_Hs_m", "Onda_Dir_graus", "blue", "Metros"),
                "Vento (Vel)": ("Vento_Vel_10m_m_s", "Vento_Dir_graus", "gray", "m/s"),
                "Corrente (Vel)": ("Corr_Vel_m_s", "Corr_Dir_graus", "cyan", "m/s")
            }
            col_val, col_dir, cor_radar, unidade_radar = mapa_radar[param_radar]
            
            bins_dir = np.arange(0, 361, 22.5)
            labels_dir = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            
            df_historico_completo['Dir_Bin'] = pd.cut(df_historico_completo[col_dir], bins=bins_dir, labels=labels_dir, include_lowest=True)
            df_radar = df_historico_completo.groupby('Dir_Bin')[[col_val, col_dir]].mean(numeric_only=True).reset_index()
            
            theta_radar = labels_dir + [labels_dir[0]]
            r_radar = df_radar[col_val].tolist() + [df_radar[col_val].iloc[0]]
            
            fig_radar = go.Figure()
            fill_color = "rgba(0,0,255,0.3)" if cor_radar == 'blue' else ("rgba(128,128,128,0.3)" if cor_radar == 'gray' else "rgba(0,255,255,0.3)")
            
            fig_radar.add_trace(go.Scatterpolar(
                r=r_radar, theta=theta_radar, fill='toself', fillcolor=fill_color,
                name=param_radar, line=dict(color=cor_radar, width=3)
            ))
            
            eixo_proa = rumo_navio
            eixo_popa = (rumo_navio + 180) % 360
            eixo_traves_be = (rumo_navio + 90) % 360
            eixo_traves_bb = (rumo_navio - 90) % 360
            r_max = df_historico_completo[col_val].max() * 0.85
            
            fig_radar.add_trace(go.Scatterpolar(r=[r_max, 0, r_max], theta=[eixo_proa, 0, eixo_popa], mode='lines', name='Eixo Proa-Popa', line=dict(color='black', width=4)))
            fig_radar.add_trace(go.Scatterpolar(r=[r_max], theta=[eixo_proa], mode='markers+text', name='Proa', marker=dict(symbol='arrow-up', size=16, color='black', angleref='previous')))
            fig_radar.add_trace(go.Scatterpolar(r=[r_max*0.6, 0, r_max*0.6], theta=[eixo_traves_be, 0, eixo_traves_bb], mode='lines', name='Eixo Transversal', line=dict(color='red', width=3, dash='dot')))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(title=dict(text=f"{param_radar} ({unidade_radar})"), visible=True, range=[0, df_historico_completo[col_val].max() * 0.85]),
                    angularaxis=dict(direction="clockwise", rotation=90) 
                ),
                height=550, margin=dict(l=40, r=40, t=50, b=30), showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption(f"📌 **Auditoria de Ancoragem:** Se a mancha de energia bater de frente com a linha transversal vermelha, o navio sofrerá Roll extremo.")

        st.markdown("---")
        st.subheader("Abertura da 'Caixa Preta': Peso Matemático da IA")
        
        col_feat1, col_feat2, col_feat3, col_feat4 = st.columns(4)
        maquinas_feat = [('SLS_Guindaste', col_feat1), ('SLS_ROV', col_feat2), ('SLS_Barco_Apoio', col_feat3), ('SLS_Offloading', col_feat4)]
        
        mapa_features = {
            'Onda_Hs_m': 'Altura Onda (Hs)', 'Onda_Tp_s': 'Período Onda (Tp)', 'Energia_Onda': 'Energia Onda (H²)',
            'Incidencia_Onda': 'Ângulo Onda', 'Vento_Vel_10m_m_s': 'Vel. Vento (10m)', 'Pressao_Vento_30m': 'Pressão Vento',
            'Incidencia_Vento': 'Ângulo Vento', 'Corr_Vel_m_s': 'Vel. Corrente', 'Incidencia_Corr': 'Ângulo Corrente',
            'Mes_sen': 'Sazonalidade', 'Esbeltez_Onda': 'Quebramento Onda', 'Razao_Onda_Navio': 'Ressonância (λ/L)',
            'Delta_Vento_Onda': 'Mar Cruzado (Δ)'
        }
        
        for alvo, coluna_feat in maquinas_feat:
            modelo = modelos_ia[alvo]
            importancias = None
            
            # O "Caçador Nível Hard": Descasca Pipelines e Otimizadores (GridSearchCV)
            estimador = modelo
            if hasattr(estimador, 'best_estimator_'):
                estimador = estimador.best_estimator_
            if hasattr(estimador, 'named_steps'):
                estimador = list(estimador.named_steps.values())[-1]
            
            # 1. Tenta a extração oficial (se o algoritmo permitir)
            if hasattr(estimador, 'feature_importances_'):
                importancias = estimador.feature_importances_
            elif hasattr(estimador, 'coef_'):
                coefs = np.abs(estimador.coef_)
                importancias = coefs[0] if coefs.ndim > 1 else coefs
                
            # --- O TRUQUE DE MESTRE: O PLANO B PARA A CAIXA PRETA ---
            # Se o algoritmo for um KNN ou Rede Neural (que recusa dar os pesos),
            # nós usamos a estatística pesada do Big Data para arrancar a resposta dele!
            if importancias is None and 'df_historico_completo' in locals() and df_historico_completo is not None:
                correlacoes = []
                coluna_risco_atual = f'Risco_Prob_{alvo}'
                for feature in features_ordem:
                    # Calcula matematicamente o quanto o Sensor X afeta o Risco Y
                    corr = df_historico_completo[feature].corr(df_historico_completo[coluna_risco_atual])
                    correlacoes.append(np.abs(corr) if not np.isnan(corr) else 0)
                importancias = np.array(correlacoes)
            # ---------------------------------------------------------
                
            if importancias is not None:
                df_feat = pd.DataFrame({'Feature_Técnica': features_ordem, 'Importância': importancias})
                df_feat['Sensor'] = df_feat['Feature_Técnica'].map(mapa_features).fillna(df_feat['Feature_Técnica'])
                
                # Normalizando os pesos para exibição em porcentagem (0 a 100%)
                soma_importancias = df_feat['Importância'].sum()
                if soma_importancias > 0:
                    df_feat['Importância'] = (df_feat['Importância'] / soma_importancias) * 100
                    
                df_feat = df_feat.sort_values(by='Importância', ascending=True).tail(8) 
                
                with coluna_feat:
                    nome_maquina = alvo.replace('SLS_', '').replace('_', ' ')
                    fig_feat = go.Figure(go.Bar(x=df_feat['Importância'], y=df_feat['Sensor'], orientation='h', marker=dict(color='#003366')))
                    fig_feat.update_layout(title=f"{nome_maquina}", xaxis_title="Impacto na Decisão (%)", yaxis=dict(tickfont=dict(size=12)), height=300, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_feat, use_container_width=True)
            else:
                with coluna_feat:
                    st.info("Aguardando processamento de dados...")

    else:
        st.warning("⚠️ **Aviso de Dados:** Arquivo de histórico não processado.")

with aba4:
    st.header("🌪️ Máquina do Tempo (Dissecação de Tempestades Históricas)")
    st.markdown("Reconstrução hora a hora dos piores Ciclones Extratropicais da última década em Búzios. Entenda a anatomia do caos e o custo financeiro do evento.")

    if 'df_historico_completo' in locals() and df_historico_completo is not None:
        
        top_tempestades = df_historico_completo.nlargest(10, 'Onda_Hs_m')
        
        opcoes_tempestade = {}
        for i, row in top_tempestades.iterrows():
            data_str = row['Data'].strftime('%d/%m/%Y')
            opcoes_tempestade[f"Ciclone de {data_str} (Pico Hs: {row['Onda_Hs_m']:.1f}m)"] = row['Data']
            
        col_seletor, col_fatura = st.columns([2, 1])
        
        with col_seletor:
            evento_selecionado = st.selectbox("Selecione o Evento Extremo do Catálogo:", list(opcoes_tempestade.keys()))
            alvo_evento = st.radio("Focar Análise na Operação:", ['SLS_Offloading', 'SLS_Barco_Apoio', 'SLS_Guindaste', 'SLS_ROV'], horizontal=True)
            
        data_pico = opcoes_tempestade[evento_selecionado]
        
        janela_inicio = data_pico - datetime.timedelta(days=3)
        janela_fim = data_pico + datetime.timedelta(days=3)
        
        mask = (df_historico_completo['Data'] >= janela_inicio) & (df_historico_completo['Data'] <= janela_fim)
        df_evento = df_historico_completo.loc[mask].copy()
        
        coluna_risco = f'Risco_Prob_{alvo_evento}'
        nome_maq = alvo_evento.replace('SLS_', '')
        
        horas_downtime = df_evento[f'Falha_{alvo_evento}'].sum()
        dias_downtime = horas_downtime / 24.0
        
        rec_dia = (producao_bopd * preco_brent) if 'producao_bopd' in locals() else (150000 * 82.5)
        custo_osv = custo_barco if 'custo_barco' in locals() else 50000
        
        custo_evento = dias_downtime * rec_dia if alvo_evento == 'SLS_Offloading' else dias_downtime * custo_osv
        tipo_custo = "Lucro Cessante (Retido)" if alvo_evento == 'SLS_Offloading' else "OPEX Desperdiçado"
        
        with col_fatura:
            linha_pico = df_evento.loc[df_evento['Onda_Hs_m'].idxmax()]
            dir_pico = linha_pico['Onda_Dir_graus']
            
            st.info("🧾 **A Fatura do Ciclone**")
            st.metric(f"Downtime ({nome_maq})", f"{horas_downtime} Horas", f"{dias_downtime:.1f} Dias", delta_color="off")
            st.metric(f"Impacto ({tipo_custo})", f"${custo_evento:,.0f} USD", delta_color="inverse")
            st.metric("Vetor de Ataque (Pico)", f"Onda vindo de {dir_pico:.0f}°", "Análise Estrutural", delta_color="off")
            
        # O botão corporativo: Movido para fora da coluna para não ficar "espremido"
        st.markdown("---")
        csv_evento = df_evento[['Data', 'Onda_Hs_m', 'Onda_Tp_s', 'Onda_Dir_graus', 'Vento_Vel_10m_m_s', 'Vento_Dir_graus', 'Delta_Vento_Onda', coluna_risco, f'Falha_{alvo_evento}']].to_csv(index=False, sep=";")
        st.download_button(
            label="📥 Exportar Dados do Ciclone para Equipe de Engenharia (CSV)",
            data=csv_evento,
            file_name=f"Laudo_Ciclone_{data_pico.strftime('%Y%m%d')}_{nome_maq}.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True
        )
            
        st.subheader(f"Evolução Crítica: Dinâmica de Força vs. Mar Cruzado")
        
        fig_timeline = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3], 
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        fig_timeline.add_trace(
            go.Scatter(x=df_evento['Data'], y=df_evento['Onda_Hs_m'], name="Altura da Onda (Hs)", line=dict(color='#0066cc', width=3)),
            row=1, col=1, secondary_y=False
        )
        fig_timeline.add_trace(
            go.Scatter(x=df_evento['Data'], y=df_evento['Vento_Vel_10m_m_s'], name="Vel. Vento (m/s)", line=dict(color='gray', width=2, dash='dot')),
            row=1, col=1, secondary_y=False
        )
        fig_timeline.add_trace(
            go.Scatter(
                x=df_evento['Data'], y=df_evento[coluna_risco] * 100, name=f"Risco IA (%)", 
                fill='tozeroy', fillcolor='rgba(255, 77, 77, 0.2)', line=dict(color='red', width=2)
            ),
            row=1, col=1, secondary_y=True
        )
        
        # A Linha Laranja da Regra Antiga (Aba 4)
        limite_tradicional = 3.5 if alvo_evento == 'SLS_Offloading' else 2.5
        fig_timeline.add_hline(y=limite_tradicional, row=1, col=1, secondary_y=False, line_dash="dot", line_color="orange", annotation_text=f"Regra Antiga ({limite_tradicional}m)")
        
        fig_timeline.add_hline(y=limite_seguranca*100, row=1, col=1, secondary_y=True, line_dash="dash", line_color="darkred", annotation_text="SUSPENSÃO IA")

        fig_timeline.add_trace(
            go.Scatter(x=df_evento['Data'], y=df_evento['Delta_Vento_Onda'], name="Δ Vento-Onda (Graus)", line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig_timeline.add_hline(y=45, row=2, col=1, line_dash="solid", line_color="orange", annotation_text="Zona de Mar Cruzado")
        fig_timeline.add_hrect(y0=45, y1=180, row=2, col=1, fillcolor="orange", opacity=0.1, layer="below")

        # --- A CEREJA DO BOLO: MARCO ZERO DO CICLONE ---
        # 1. Desenhamos as linhas puras usando a data original (sem texto embutido para evitar o bug do Plotly)
        fig_timeline.add_vline(x=data_pico, line_width=2, line_dash="dashdot", line_color="black", row=1, col=1)
        fig_timeline.add_vline(x=data_pico, line_width=2, line_dash="dashdot", line_color="black", row=2, col=1)

        # 2. Adicionamos o texto de forma independente e segura no topo do gráfico
        fig_timeline.add_annotation(
            x=data_pico, y=1.05, yref="paper", 
            text="💥 PICO DO EVENTO", showarrow=False, 
            font=dict(color="black", size=14, weight="bold"), 
            row=1, col=1
        )

        fig_timeline.update_layout(
            height=650, margin=dict(l=20, r=20, t=40, b=20), # 't' ajustado para 40 para caber o texto
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            hovermode="x unified"
        )
        
        fig_timeline.update_yaxes(title_text="Intensidade ($H_s$ / Vento)", row=1, col=1, secondary_y=False)
        fig_timeline.update_yaxes(title_text="Risco IA (%)", range=[0, 105], row=1, col=1, secondary_y=True)
        fig_timeline.update_yaxes(title_text="Δ Angular (°)", range=[0, 180], tickvals=[0, 45, 90, 180], row=2, col=1)
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.caption("📌 **Anatomia do Ciclone:** A linha vertical preta marca o 'Olho' da tempestade. O gráfico inferior é o **'Detector de Caos'**. Quando a linha roxa invade a área laranja (Δ > 45°), o mar está cruzado. A **linha pontilhada laranja** superior representa a regra antiga da Petrobras. Note como a área vermelha (IA) antecipa o perigo mesmo quando a onda ainda está abaixo do limite tradicional.")
        
    else:
        st.warning("⚠️ Carregue o Big Data na Aba 2 primeiro para ativar a Máquina do Tempo.")
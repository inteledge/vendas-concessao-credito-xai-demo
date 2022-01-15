import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

##########
#Imports
##########
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

from evalml.model_understanding.prediction_explanations import explain_predictions

##########
#Configuração do Streamlit
##########
st.set_page_config(page_title='Inteledge - Simulador de Crédito', page_icon="💡", layout="centered", initial_sidebar_state="auto", menu_items=None)

##########
#Funções para as previsões e para a organização da pãgina
##########
@st.cache(allow_output_mutation=True)
def get_pickles():
	logical_types = pickle.load(open('sample.pkl', 'rb'))
	best_pipeline, expected_value = pickle.load(open('model.pkl', 'rb'))
	
	return best_pipeline, logical_types, expected_value

@st.cache(allow_output_mutation=True)
def get_samples(target):
	# carregando uma amostra da base de dados
	df = pickle.load(open('df_resampled.pkl', 'rb'))
	columns = [target] + df.drop(target, axis=1).columns.tolist()

	# carregando as predições
	df[target] = best_pipeline.predict(df.drop(target, axis=1))
	df = df.replace({target: {1: 'Aprovado', 0: 'Reprovado'}})
	
	df_negados = df[df[target]=='Reprovado'].tail(5).reset_index(drop=True)
	df_negados = df_negados[columns]

	df_aprovados = df[df[target]=='Aprovado'].tail(5).reset_index(drop=True)
	df_aprovados = df_aprovados[columns]
	
	return df, df_negados, df_aprovados

def plot_importances(best_pipeline, df):
	# predictions
	pred = best_pipeline.predict(df).values[0]
	pred = 'Aprovado' if pred == 1 else 'Reprovado'
	pred_proba = best_pipeline.predict_proba(df).values[0]

	starting_value = expected_value*100

	df_plot = explain_predictions(pipeline=best_pipeline, input_features=df.reset_index(drop=True),
							y=None, top_k_features=len(df.columns), indices_to_explain=[0],
							include_explainer_values=True, output_format='dataframe')

	if np.argmax(pred_proba) == 1:
		df_plot['quantitative_explanation'] = df_plot['quantitative_explanation']*100
	else:
		starting_value = 100-starting_value
		df_plot['quantitative_explanation'] = df_plot['quantitative_explanation']*-100

	df_plot = df_plot.sort_values(by='quantitative_explanation')
	df_plot['Soma'] = starting_value+df_plot['quantitative_explanation'].cumsum()
	df_plot = df_plot.sort_values(by='quantitative_explanation')
	df_plot['Influencia para este resultado?'] = df_plot['quantitative_explanation']<0
	df_plot = df_plot.round(2)

	col_names = []
	for col in df_plot['feature_names'].values:
		col_names.append(f'{col}<br><em>({df[col].values[0]})</em>')

	fig_xai = go.Figure(go.Waterfall(
		name='Projeção',
		base=0,
		orientation="h",
		y=['Inicial'] + col_names + ['Final'],
		x=[starting_value] + df_plot['quantitative_explanation'].values.tolist() + [0],
		measure=['absolute'] + ['relative']*len(df_plot) + ['total'],
		text=[None] + [f'{x:.1f}%' for x in df_plot['Soma'].values] + [None],
		#textposition = "outside",
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
	))

	#fig_xai.update_xaxes(range=[max(0, df_plot['Soma'].min()*0.7),
	#						min(100, df_plot['Soma'].max()*1.2)])

	fig_xai.update_layout(
	title=f'Principais influenciadores para o resultado:<br>(Previsão: <b>{pred}</b>, com {round(100*pred_proba.max(),1)}% de certeza)',
	showlegend = False,
	width=420,
	height=1100
	)

	return fig_xai, pred

##########
#Preparando o simulador
##########
# carregando o modelo preditivo
best_pipeline, logical_types, expected_value = get_pickles()

# carregando uma amostra da base de dados
target = 'Aprovar o crédito?'
df, df_negados, df_aprovados = get_samples(target)

##########
#Seção 1 - Histõrico
##########
col1, _, _ = st.columns(3)
with col1:
	st.image('inteledge.png')

st.title('Simulador de Crédito')
st.markdown('Inteligência artificial para previsão de risco de crédito para novos clientes construído para uma base de dados fictícia pela Inteledge para você testar. Ficou interessado em fazer algo parecido para o seu negócio? Entre em contato conosco em @inteledge.lab no [Instagram](https://instagram.com/inteledge.lab) ou no [LinkedIn](https://linkedin.com/inteledge.lab)!')
st.write('Aqui, mostramos a previsão por cliente e como o nosso algoritmo chegou à conclusão para cada caso.')
st.markdown('Confira também [algumas análises que fizemos para esta base de dados](https://share.streamlit.io/wmonteiro92/vendas-concessao-credito-analise-demo/main/exploration.py).')

st.header('Últimas previsões')

st.write('Últimos 5 casos reprovados:')
st.dataframe(df_negados)

st.write('Últimos 5 casos aprovados:')
st.dataframe(df_aprovados)

##########
#Seção 2 - Simulador
##########
st.header('Simulador de novos clientes')
st.markdown('Veja como seriam as previsões para novos clientes. Teste diferentes configurações e veja ao lado como o algoritmo chegou a esta conclusão. As previsões atualizam **em tempo real**.')
st.write('Do ponto de vista de Ciência de Dados, nenhum algoritmo será 100% preciso. Por outro lado, saber como ele funciona (e como ele chega às suas conclusões) nos dá um nível de confiança maior sobre a sua lógica.')
col1, col2 = st.columns(2)

with col1:
	# variáveis 
	idade = st.slider('Idade',
		int(df['Idade'].min()), int(df['Idade'].max()), int(df_negados['Idade'].iloc[0]))

	valor_pedido = st.slider('Valor pedido',
		int(df['Valor pedido'].min()), int(df['Valor pedido'].max()),
		int(df_negados['Valor pedido'].iloc[0]))
		 
	num_parcelas = st.slider('Número de parcelas',
		int(df['Número de parcelas'].min()), int(df['Número de parcelas'].max()),
		int(df_negados['Número de parcelas'].iloc[0]))

	tempo_residencia = st.slider('Tempo morando na residência atual',
		int(df['Tempo morando na residência atual'].min()),
		int(df['Tempo morando na residência atual'].max()),
		int(df_negados['Tempo morando na residência atual'].iloc[0]))
	   
	num_emprestimos = st.slider('Número de empréstimos passados',
		int(df['Número de empréstimos passados'].min()),
		int(df['Número de empréstimos passados'].max()),
		int(df_negados['Número de empréstimos passados'].iloc[0]))
		
	num_referencias = st.slider('Número de referências',
		int(df['Número de referências'].min()), int(df['Número de referências'].max()),
		int(df_negados['Número de referências'].iloc[0]))
		
	saldo_conta = st.selectbox('Saldo na conta',
		('Negativo', 'Entre 0 e 10000', 'Entre 10001 e 80000', 'Acima de 80001'))

	historico_emprestimos = st.selectbox('Histórico de empréstimos',
		('Atrasou mais de 3x no passado', 'Atrasou até 3x no passado', 
		'Ainda possui dívidas em outros bancos', 'Sempre pagou em dia em outros bancos',
		'Sempre pagou em dia neste banco'))

	motivo_emprestimo = st.selectbox('Motivo do empréstimo',
		('Carro usado', 'Carro novo', 'Móveis', 'Eletrônicos',
		'Eletrodomésticos', 'Material de construção', 'Educação',
		'Férias', 'Mão-de-obra', 'Casa', 'Outros'), 1)

	investimento_em_conta = st.selectbox('Investimentos em conta',
		('Menos de 5000', 'Entre 5001 e 10000', 'Entre 10001 e 50000',
		'Entre 50001 e 100000', 'Mais de 100001'))
		
	tempo_emprego = st.selectbox('Tempo no emprego atual',
		('Desempregado', 'Menos de 1 ano', 'Entre 1 e 3 anos',
		'Entre 3 e 5 anos', 'Mais de 5 anos'))

	situacao_familiar = st.selectbox('Situação familiar',
		('Solteiro(a), sem filhos', 'Solteiro(a), com filhos',
		'Casado(a), sem filhos', 'Casado(a), com filhos', 'Viúvo(a)'))

	avalista_fiador = st.selectbox('Possui avalista/fiador?',
		('Não', 'Avalista', 'Fiador'))

	outros_bens = st.selectbox('Outros bens',
		('Sem bens', 'Carro', 'Casa própria', 'Outros'))

	outros_emprestimos = st.selectbox('Outros empréstimos',
		('Nunca teve', 'Teve em outras empresas', 'Teve neste banco'))

	residencia_atual = st.selectbox('Residência atual',
		('Aluguel', 'Casa própria', 'Outros'))

	trabalho_atual = st.selectbox('Trabalho atual',
		('Desempregado/sem comprovação', 'Operacional',
		'Analista', 'Gestor/Especialista/Empresário'))

	possui_cartao = st.checkbox('Possui cartão de crédito')

	estrangeiro = st.checkbox('É estrangeiro?')

with col2:
	# inference
	df_inference = pd.DataFrame([[idade, valor_pedido, num_parcelas, tempo_residencia,
		num_emprestimos, num_referencias, saldo_conta, historico_emprestimos,
		motivo_emprestimo, investimento_em_conta, tempo_emprego, situacao_familiar, 
		avalista_fiador, outros_bens, outros_emprestimos, residencia_atual,
		trabalho_atual, possui_cartao, estrangeiro]],
		columns=['Idade', 'Valor pedido', 'Número de parcelas', 'Tempo morando na residência atual',
		'Número de empréstimos passados', 'Número de referências', 'Saldo na conta',
		'Histórico de empréstimos', 'Motivo do empréstimo', 'Investimentos em conta',
		'Tempo no emprego atual', 'Situação familiar', 'Possui avalista/fiador?',
		'Outros bens', 'Outros empréstimos', 'Residência atual', 'Trabalho atual',
		'Possui cartão de crédito', 'É estrangeiro?'])
	df_inference = df_inference[logical_types.keys()]
	df_inference.ww.init()
	df_inference.ww.set_types(logical_types=logical_types)

	fig_xai, predicao = plot_importances(best_pipeline, df_inference)
	st.plotly_chart(fig_xai)
    
st.markdown('Siga-nos no [Instagram](https://instagram.com/inteledge.lab) e no [LinkedIn](https://linkedin.com/inteledge.lab)!')
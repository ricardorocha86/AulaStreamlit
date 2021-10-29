import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

modelo1 = load_model('meu-modelo-para-charges')
modelo2 = load_model('meu-modelo-para-smoker')

paginas = ['Home', 'Modelo Custos', 'Modelo Fumante']

pagina = st.sidebar.radio('Navegue por aqui:', paginas)

if pagina == 'Home':
	st.title('Meus Modelos em Produção :gem:')

	st.write('Navegue pelo menu na barra lateral para escolher dentre os modelos disponíveis nessa aplicação Web')

if pagina == 'Modelo Custos':


	st.title('Modelo para Previsão de Custos de Seguro')

	idade = st.number_input('Idade', 18, 65, 30)
	sexo = st.selectbox("Sexo", ['Masculino', 'Feminino'])
	imc = st.number_input('Índice de Massa Corporal', 15, 54, 24)
	criancas = st.selectbox("Quantidade de dependentes", [0, 1, 2, 3, 4, 5])
	fumante = st.selectbox("É fumante?", ['yes', 'no'])
	regiao = st.selectbox("Região em que mora", 
									  ['southeast', 'southwest', 'northeast', 'northwest'])

	sex = 'male' if sexo == 'Masculino' else 'female'

	dados0 = {'age': [idade], 'sex': [sex], 'bmi': [imc], 'children': [criancas], 'smoker': [fumante], 'region': [regiao]}
	dados = pd.DataFrame(dados0)

	st.markdown('---')

	if st.button('EXECUTAR MODELO'):
		pred = float(predict_model(modelo1, data = dados)['Label'].round(2))
		saida = 'O valor predito é de ${:.2f}'.format(pred)
		st.subheader(saida)





if pagina == 'Modelo Fumante':
	st.title('Modelo para Previsão de Possíveis Fraudadores')


	idade = st.number_input('Idade', 18, 65, 30)
	sexo = st.selectbox("Sexo", ['Masculino', 'Feminino'])
	imc = st.number_input('Índice de Massa Corporal', 15, 54, 24)
	criancas = st.selectbox("Quantidade de dependentes", [0, 1, 2, 3, 4, 5])
	custos = st.slider('Custos', 1000, 50000, 10000, 1000)
	regiao = st.selectbox("Região em que mora", 
									  ['southeast', 'southwest', 'northeast', 'northwest'])

	sex = 'male' if sexo == 'Masculino' else 'female'

	dados0 = {'age': [idade], 'sex': [sex], 'bmi': [imc], 'children': [criancas], 'charges': [custos], 'region': [regiao]}
	dados = pd.DataFrame(dados0)

	st.markdown('---')

	if st.button('EXECUTAR MODELO'):
		pred = predict_model(modelo2, data = dados)['Label']
		st.write(pred)
		#saida = 'O valor predito é de ${:.2f}'.format(pred)
		#st.subheader(saida)

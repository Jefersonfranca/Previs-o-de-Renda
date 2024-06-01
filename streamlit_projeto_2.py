import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import io

import numpy as np

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

sns.set(context='talk', style='ticks')


st.set_page_config(
    page_title="Projeto #02 | Previsão de renda",
    page_icon="https://raw.githubusercontent.com/Jefersonfranca/Previs-o-de-Renda/main/imagens/favicon.ico",
    layout="wide",
    initial_sidebar_state="auto",
)

st.sidebar.markdown('''
<div style="text-align:center">
<img src="https://raw.githubusercontent.com/Jefersonfranca/Previs-o-de-Renda/main/imagens/newebac_logo_black_half.png" alt="ebac-logo" width=50%>
</div>

# **Profissão: Cientista de Dados**
### **Projeto #02** | Previsão de renda
<div>

**Jeferson França**   
[  LinkedIn](https://www.linkedin.com/in/jeferson-frança-bastos/)<br>
### [Repositório no Github](https://github.com/Jefersonfranca/Previs-o-de-Renda/)
<div>

---
''', unsafe_allow_html=True)

st.markdown('''
<div style="text-align:center">
            
<h1>Análise exploratória da previsão de renda</h1>
''', unsafe_allow_html=True)

st.divider()

st.markdown('''
## Etapa 1 CRISP - DM: Entendimento do negócio <a name="1"></a>
''', unsafe_allow_html=True)

st.markdown('''
A análise detalhada de dados são fundamentais para qualquer instituição bancária que deseja melhorar seu relacionamento com os clientes, oferecer produtos mais relevantes e gerir riscos de forma eficiente. 

Quanto mais informações disponíveis permite uma visão holística dos clientes, possibilitando decisões mais estratégicas e eficazes em todas as áreas do banco.
''')

st.markdown('''
## Etapa 2 Crisp-DM: Entendimento dos dados<a name="2"></a>
''', unsafe_allow_html=True)

st.markdown('''
Este banco de dados contém informações detalhadas sobre os clientes de uma instituição bancária, proporcionando uma visão abrangente de diversos aspectos sociodemográficos e financeiros. 
As tabelas estão estruturadas para armazenar dados relevantes que podem ser utilizados para análises e insights aprofundados.

Esta ferramenta é poderosa para a análise de perfil dos clientes, permitindo ao banco realizar segmentações, identificar padrões e comportamentos, e tomar decisões mais informadas sobre serviços e produtos financeiros personalizados. 
Além disso, pode ser utilizado para modelar risco de crédito, prever tendências e desenvolver estratégias de marketing direcionadas.
''')

st.markdown('''
### Dicionário de dados <a name="dicionario"></a>

| Variável                | Descrição                                           | Tipo         |
| ----------------------- |:---------------------------------------------------:| ------------:|
| data_ref                |  Data de referência de coleta das variáveis         | object|
| id_cliente              |  Código identificador exclusivo do cliente          | int64|
| sexo                    |  Sexo do cliente                                    | object|
| posse_de_veiculo        |  Indica se o cliente possui veículo                 | bool|
| posse_de_imovel         |  Indica se o cliente possui imóvel                  | bool|
| qtd_filhos              |  Quantidade de filhos do cliente                    | int64|
| tipo_renda              |  Tipo de renda do cliente                           | object|
| educacao                |  Grau de instrução do cliente                       | object|
| estado_civil            |  Estado civil do cliente                            | object|
| tipo_residencia         |  Tipo de residência do cliente (própria, alugada etc)| object|
| idade                   |  Idade do cliente                                   | int64|
| tempo_emprego           |  Tempo no emprego atual                             | float64|
| qt_pessoas_residencia   |  Quantidade de pessoas que moram na residência      | float64|
| renda                   |  Renda em reais                                     | float64|
''', unsafe_allow_html=True)


st.markdown('''
### Carregando os dados <a name="dados"></a>
''', unsafe_allow_html=True)


filepath = 'https://raw.githubusercontent.com/Jefersonfranca/EBAC_Curso_Cientista_de_Dados/main/M%C3%B3dulo%2013%20Regress%C3%A3o%20II/database/previsao_de_renda.csv'
renda = pd.read_csv(filepath_or_buffer=filepath)

buffer = io.StringIO()
renda.info(buf=buffer)
st.text(buffer.getvalue())
st.dataframe(renda)

st.markdown('''
### Entendimento dos dados - Univariada <a name="univariada"></a>
''', unsafe_allow_html=True)

with st.expander("Pandas Profiling – Relatório interativo para análise exploratória de dados", expanded=True):
    prof = ProfileReport(df=renda,
                         minimal=False,
                         explorative=True,
                         dark_mode=True,
                         orange_mode=True)
    # st.components.v1.html(prof.to_html(), height=600, scrolling=True)
    st_profile_report(prof)

st.markdown('''
####  Estatísticas descritivas das variáveis quantitativas <a name="describe"></a>
''', unsafe_allow_html=True)

st.markdown('''
  A análise univariada inicial revelou que algumas variáveis do dataframe não são úteis e serão removidas na etapa de limpeza dos dados. Ao mesmo tempo, essa análise destacou a importância de outras variáveis.

Variáveis como **Unnamed: 0**, **data\_ref** e **id\_cliente** não contribuem para a obtenção de insights significativos e, portanto, serão descartadas. Por outro lado, as demais variáveis apresentam potencial para fornecer informações valiosas e merecem uma análise mais aprofundada.

Já no alerta mostrou alugmas variáveis que devem ser analisadas se serão ou não removidas na etapa de limpeza.
''')

st.markdown('''
### Entendimento dos dados - Bivariadas <a name="bivariada"></a>
''', unsafe_allow_html=True)

st.markdown('''
#### Matriz de correlação <a name="correlacao"></a>
''', unsafe_allow_html=True)

st.write((renda
          .iloc[:, 3:]
          .corr(numeric_only=True)
          .tail(n=6)
          ))

st.markdown('''
#### Matriz de dispersão <a name="dispersao"></a>
''', unsafe_allow_html=True)

sns.pairplot(data=renda,
             hue='tipo_renda',
             vars=['qtd_filhos',
                   'idade',
                   'tempo_emprego',
                   'qt_pessoas_residencia',
                   'renda'],
             diag_kind='hist')
st.pyplot(plt)

st.markdown('Ao analisar o pairplot, que consiste na matriz de dispersão, é possível identificar alguns outliers na variável renda, os quais podem afetar o resultado da análise de tendência, apesar de ocorrerem com baixa frequência. Além disso, é observada uma baixa correlação entre praticamente todas as variáveis quantitativas.')

st.markdown('''
##### Clustermap <a name="clustermap"></a>
''', unsafe_allow_html=True)

cmap = sns.diverging_palette(h_neg=100,
                             h_pos=359,
                             as_cmap=True,
                             sep=1,
                             center='light')
ax = sns.clustermap(data=renda.corr(numeric_only=True),
                    figsize=(10, 10),
                    center=0,
                    cmap=cmap)
plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=45)
st.pyplot(plt)

st.markdown('Com o *clustermap*, é possível reforçar novamente os resultados de baixa correlação com a variável `renda`. Apenas a variável `tempo_emprego` apresenta um índice considerável para análise. Além disso, foram apresentadas duas variáveis booleanas, `posse_de_imovel` e `posse_de_veiculo`, mas que também possuem baixo índice de correlação com renda.')

st.markdown('''
#####  Linha de tendência <a name="tendencia"></a>
''', unsafe_allow_html=True)

plt.figure(figsize=(16, 9))
sns.scatterplot(x='tempo_emprego',
                y='renda',
                hue='tipo_renda',
                size='idade',
                data=renda,
                alpha=0.4)
sns.regplot(x='tempo_emprego',
            y='renda',
            data=renda,
            scatter=False,
            color='.3')
st.pyplot(plt)

st.markdown('Embora a correlação entre a variável `tempo_emprego` e a variável `renda` não seja tão alta, é possível identificar facilmente a covariância positiva com a inclinação da linha de tendência.')

st.markdown('''
#### Análise das variáveis qualitativas <a name="qualitativas"></a>
''', unsafe_allow_html=True)


with st.expander("Análise de relevância preditiva com variáveis booleanas", expanded=True):
    plt.rc('figure', figsize=(12, 4))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.pointplot(x='posse_de_imovel',
                  y='renda',
                  data=renda,
                  dodge=True,
                  ax=axes[0])
    sns.pointplot(x='posse_de_veiculo',
                  y='renda',
                  data=renda,
                  dodge=True,
                  ax=axes[1])
    st.pyplot(plt)

    st.markdown('Ao comparar os gráficos acima, nota-se que a variável `posse_de_veículo` apresenta maior relevância na predição de renda, evidenciada pela maior distância entre os intervalos de confiança para aqueles que possuem e não possuem veículo, ao contrário da variável `posse_de_imóvel` que não apresenta diferença significativa entre as possíveis condições de posse imobiliária.')


with st.expander("Análise das variáveis qualitativas ao longo do tempo", expanded=True):
    renda['data_ref'] = pd.to_datetime(arg=renda['data_ref'])
    qualitativas = renda.select_dtypes(include=['object', 'boolean']).columns
    plt.rc('figure', figsize=(16, 4))
    for col in qualitativas:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.subplots_adjust(wspace=.6)
        tick_labels = renda['data_ref'].map(
            lambda x: x.strftime('%b/%Y')).unique()
        # barras empilhadas:
        renda_crosstab = pd.crosstab(index=renda['data_ref'],
                                     columns=renda[col],
                                     normalize='index')
        ax0 = renda_crosstab.plot.bar(stacked=True,
                                      ax=axes[0])
        ax0.set_xticklabels(labels=tick_labels, rotation=45)
        axes[0].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
        # perfis médios no tempo:
        ax1 = sns.pointplot(x='data_ref', y='renda', hue=col,
                            data=renda, dodge=True, errorbar=('ci', 95), ax=axes[1])
        ax1.set_xticklabels(labels=tick_labels, rotation=45)
        axes[1].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
        st.pyplot(plt)

st.markdown('''
## Etapa 3 Crisp-DM: Preparação dos dados<a name="3"></a>
''', unsafe_allow_html=True)

renda = pd.read_csv(filepath_or_buffer=filepath, index_col=0)

renda.drop(columns='data_ref', inplace=True)
renda.dropna(inplace=True)
st.table(pd.DataFrame(index=renda.nunique().index,
                      data={'tipos_dados': renda.dtypes,
                            'qtd_valores': renda.notna().sum(),
                            'qtd_categorias': renda.nunique().values}))

#renda['sexo'] = renda.sexo.map({'F': 1, 'M':0})
#renda.sexo.unique()

with st.expander("Conversão das variáveis categóricas em variáveis numéricas (dummies)", expanded=True):
    renda_dummies = pd.get_dummies(data=renda)
    buffer = io.StringIO()
    renda_dummies.info(buf=buffer)
    st.text(buffer.getvalue())

    st.table((renda_dummies.corr()['renda']
              .sort_values(ascending=False)
              .to_frame()
              .reset_index()
              .rename(columns={'index': 'var',
                               'renda': 'corr'})
              .style.bar(color=['darkred', 'darkgreen'], align=0)
              ))

st.markdown('''
## Etapa 4 Crisp-DM: Modelagem <a name="4"></a>
''', unsafe_allow_html=True)

st.markdown('A técnica escolhida foi o DecisionTreeRegressor, devido à sua capacidade de lidar com problemas de regressão, como a previsão de renda dos clientes. Além disso, árvores de decisão são fáceis de interpretar e permitem a identificação dos atributos mais relevantes para a previsão da variável-alvo, tornando-a uma boa escolha para o projeto.')

st.markdown('''
### Divisão da base em treino e teste <a name="train_test"></a>
''', unsafe_allow_html=True)

X = renda_dummies.drop(columns='renda')
y = renda_dummies['renda']
st.write('Quantidade de linhas e colunas de X:', X.shape)
st.write('Quantidade de linhas de y:', len(y))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
st.write('X_train:', X_train.shape)
st.write('X_test:', X_test.shape)
st.write('y_train:', y_train.shape)
st.write('y_test:', y_test.shape)

st.markdown('''
### Seleção de hiperparâmetros do modelo com for loop <a name="for_loop"></a>
''', unsafe_allow_html=True)

score = pd.DataFrame(columns=['max_depth', 'min_samples_leaf', 'score'])
for x in range(1, 21):
    for y in range(1, 31):
        reg_tree = DecisionTreeRegressor(random_state=42,
                                         max_depth=x,
                                         min_samples_leaf=y)
        reg_tree.fit(X_train, y_train)
        score = pd.concat(objs=[score,
                                pd.DataFrame({'max_depth': [x],
                                              'min_samples_leaf': [y],
                                              'score': [reg_tree.score(X=X_test,
                                                                       y=y_test)]})],
                          axis=0,
                          ignore_index=True)
st.dataframe(score.sort_values(by='score', ascending=False))

st.markdown('''
### Rodando o modelo <a name="rodando"></a>
''', unsafe_allow_html=True)

reg_tree = DecisionTreeRegressor(random_state=42,
                                 max_depth=8,
                                 min_samples_leaf=4)
# reg_tree.fit(X_train, y_train)
st.text(reg_tree.fit(X_train, y_train))

with st.expander("Visualização gráfica da árvore com plot_tree", expanded=True):
    plt.figure(figsize=(18, 9))
    tree.plot_tree(decision_tree=reg_tree,
                   feature_names=X.columns,
                   filled=True)
    st.pyplot(plt)

with st.expander("Visualização impressão da árvore", expanded=False):
    text_tree_print = tree.export_text(decision_tree=reg_tree)
    st.text(text_tree_print)

st.markdown('''
## Etapa 5 Crisp-DM: Avaliação dos resultados <a name="5"></a>
''', unsafe_allow_html=True)

r2_train = reg_tree.score(X=X_train, y=y_train)
r2_test = reg_tree.score(X=X_test, y=y_test)
template = 'O coeficiente de determinação R² da árvore com profundidade = {0} para a base de {1} é: {2:.2f}'
st.write(template.format(reg_tree.get_depth(),
                         'treino',
                         r2_train)
         .replace(".", ","))
st.write(template.format(reg_tree.get_depth(),
                         'teste',
                         r2_test)
         .replace(".", ","))


renda['renda_predict'] = np.round(reg_tree.predict(X), 2)
st.dataframe(renda[['renda', 'renda_predict']])

st.markdown('''
## Etapa 6 Crisp-DM: Implantação <a name="6"></a>
''', unsafe_allow_html=True)

st.markdown('# Use o simulador na barra lateral.')

st.sidebar.header("Simulando da previsão de renda")
with st.sidebar.form("my_form"):
    st.subheader("Preencha os campos abaixo:")
    
    #renda['sexo'] = renda.sexo.map({1: 'F', 0:'M'})
    sexo = st.radio("Sexo", ('M', 'F'))
    veiculo = st.checkbox("Posse de veículo")
    imovel = st.checkbox("Posse de imóvel")
    filhos = st.number_input("Quantidade de filhos", 0, 15)
    tiporenda = st.selectbox("Tipo de renda", [
                             'Sem renda', 'Empresário', 'Assalariado', 'Servidor público', 'Pensionista', 'Bolsista'])
    if tiporenda == 'Sem renda':
        tiporenda = None
    educacao = st.selectbox("Educação", [
                            'Primário', 'Secundário', 'Superior incompleto', 'Superior completo', 'Pós graduação'])
    estadocivil = st.selectbox(
        "Estado civil", ['Solteiro', 'União', 'Casado', 'Separado', 'Viúvo'])
    residencia = st.selectbox("Tipo de residência", [
                              'Casa', 'Governamental', 'Com os pais', 'Aluguel', 'Estúdio', 'Comunitário'])
    idade = st.slider("Idade", 18, 100)
    tempoemprego = st.slider("Tempo de emprego", 0, 50)
    qtdpessoasresidencia = st.number_input(
        "Quantidade de pessoas na residência", 1, 15)

    submitted = st.form_submit_button("Simular")
    if submitted:
        entrada = pd.DataFrame([{'sexo': sexo,
                                 'posse_de_veiculo': veiculo,
                                 'posse_de_imovel': imovel,
                                 'qtd_filhos': filhos,
                                 'tipo_renda': tiporenda,
                                 'educacao': educacao,
                                 'estado_civil': estadocivil,
                                 'tipo_residencia': residencia,
                                 'idade': idade,
                                 'tempo_emprego': tempoemprego,
                                 'qt_pessoas_residencia': qtdpessoasresidencia}])
        entrada = pd.concat([X, pd.get_dummies(entrada)]
                            ).fillna(value=0).tail(1)
        st.write(
            f"Renda estimada: R${str(np.round(reg_tree.predict(entrada).item(), 2)).replace('.', ',')}")

'---'
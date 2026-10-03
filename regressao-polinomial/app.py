import gradio as gr
import joblib
import pandas as pd

modelo = joblib.load('modelo.pkl')


def prever(tempo_de_experiencia, numero_de_vendas, fator_sazonal):
    dados = pd.DataFrame({
        'tempo_de_experiencia': [tempo_de_experiencia],
        'numero_de_vendas': [numero_de_vendas],
        'fator_sazonal': [fator_sazonal],
    })
    return float(modelo.predict(dados)[0])


app = gr.Interface(
    fn=prever,
    inputs=[
        gr.Number(label='Tempo de experiência (meses)'),
        gr.Number(label='Número de vendas'),
        gr.Slider(1, 10, step=1, label='Fator sazonal'),
    ],
    outputs=gr.Number(label='Receita prevista (R$)'),
)

app.launch()

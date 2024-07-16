from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Carregar o modelo treinado
model_path = 'mlp_model_chagas.pkl'
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obter dados do formulário
        data = [float(request.form.get(var, 0)) for var in ['EDEMA', 'MENINGOE', 'POLIADENO', 'FEBRE', 'HEPATOME', 'SINAIS_ICC', 'ARRITMIAS', 'ASTENIA', 'ESPLENOM', 'CHAGOMA']]
        data = np.array(data).reshape(1, -1)

        # Fazer a predição
        prob = model.predict_proba(data)[0, 1]

        # Gerar gráfico
        fig, ax = plt.subplots()
        ax.bar(['Probabilidade de Doença de Chagas'], [prob])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilidade')

        # Converter gráfico para imagem base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', prob=prob, plot_url=plot_url)

    return render_template('index.html', prob=None, plot_url=None)

if __name__ == '__main__':
    app.run(debug=True)
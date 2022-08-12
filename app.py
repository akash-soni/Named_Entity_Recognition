from flask import Flask, render_template, request, jsonify,redirect, url_for, Response
from flask_cors import CORS,cross_origin
from ner.components.predictions import PredictionClassifier
from ner.pipline.predict_pipeline import Prediction_Pipeline
from ner.config.configurations import Configuration
from ner.components.predictions import PredictionClassifier

app = Flask(__name__)  # initialising the flask app with the name 'app'


@app.route('/',methods=['POST','GET']) # route with allowed methods as POST and GET
@cross_origin()
def index():
    if request.method == 'POST':
        text = request.form['input_text']
        print(text)
        config = Configuration()
        prediction_configuration = config.get_model_predict_pipeline_config()

        prediction_obj = PredictionClassifier(prediction_configuration, text)
        pred_tags = prediction_obj.prediction()
        return render_template('index.html', input_text=text, pred_tags= pred_tags )
    return render_template('index.html')


if __name__ == "__main__":
    app.run(port=7000, debug=True)
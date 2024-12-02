from flask import Flask, render_template, request

from src.pipe.prediction_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        age = request.form.get('age')
        sex = request.form.get('sex')
        chestpaintype = request.form.get('chestpaintype')
        restingbp = request.form.get('restingbp')
        cholesterol = request.form.get('cholesterol')
        fastingbs = request.form.get('fastingbs')
        restingecg = request.form.get('restingecg')
        maxhr = request.form.get('maxhr')
        exerciseangina = request.form.get('exerciseangina')
        oldpeak = request.form.get('oldpeak')
        st_slope = request.form.get('st_slope')

        customdata = CustomData(
            age=age,
            sex=sex,
            chestpaintype=chestpaintype,
            restingbp=restingbp,
            cholesterol=cholesterol,
            fastingbs=fastingbs,
            restingecg=restingecg,
            maxhr=maxhr,
            exerciseengina=exerciseangina,
            oldpeak=oldpeak,
            st_slope=st_slope
        )
        dataframe = customdata.conver_data_to_dataframe()

        print(dataframe.head())

        prediction_pipe = PredictionPipeline()
        prediction = prediction_pipe.predict(dataframe)

        return render_template('form.html', prediction=prediction)

    return render_template('form.html') 

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
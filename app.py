from flask import Flask, request, render_template
import pickle
import pandas as pd
import joblib
app = Flask(__name__, template_folder="./")

# Завантажте збережену модель Pickle
# model = joblib.load('logistic_regression_model.pkl')
try:
    model = joblib.load('k_bp_model.pkl')
except Exception as e:
    print("Помилка завантаження моделі:", str(e))


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Отримайте дані з веб-форми
        age = float(request.form['age'])
        gender = request.form['gender']
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        residence_type = request.form['Residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])

        # # Зробіть передбачення за допомогою моделі
        data = pd.DataFrame([[gender, age, hypertension, heart_disease,
                              ever_married, work_type, residence_type, avg_glucose_level, bmi]],
                            columns=['gender', 'age', 'hypertension', 'heart_disease',
                                     'ever_married', 'work_type', 'Residence_type',
                                     'avg_glucose_level', 'bmi'])

        # if gender == 'Male':
        #     data['gender'] = 1
        # else:
        #     data['gender'] = 0
        # if ever_married == 'Yes':
        #     data['ever_married'] = 1
        # else:
        #     data['ever_married'] = 0
        # if work_type == 'children':
        #     data['work_type'] = 4
        # elif work_type == 'Private':
        #     data['work_type'] = 0
        # elif work_type == 'Never_worked':
        #     data['work_type'] = 1

        # if residence_type == 'Urban':
        #     data['Residence_type'] = 1
        # else:
        #     data['Residence_type'] = 0
        prediction = model.predict(data)[0]
        # prediction = model.predict(data)
        probabilities = model.predict_proba(data)[0]
        # probabilities = model.predict_proba(data)
        confidence = max(probabilities) * 100
        # Поверніть результат передбачення на сторінку
        return render_template('result.html', prediction=prediction, confidence=confidence)
    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)

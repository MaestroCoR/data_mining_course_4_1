from flask import Flask, request, render_template
import pandas as pd
import joblib
app = Flask(__name__, template_folder="./")

# Завантаження збереженої моделі

try:
    # model = joblib.load('logistic_regression_model.pkl')
    model = joblib.load('k_bp_model.pkl')
except Exception as e:
    print("Помилка завантаження моделі:", str(e))


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Отримання даних з веб-форми
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])

        # передбачення за допомогою моделі
        data = pd.DataFrame([[gender, age, hypertension, heart_disease,
                              ever_married, work_type, residence_type, avg_glucose_level, bmi]],
                            columns=['gender', 'age', 'hypertension', 'heart_disease',
                                     'ever_married', 'work_type', 'Residence_type',
                                     'avg_glucose_level', 'bmi'])

        print(data)
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        confidence = max(probabilities) * 100
        # Округлимо до 2 знаків після коми
        confidence = round(confidence, 2)
        # Повернути результат передбачення на сторінку
        return render_template('result.html', prediction=prediction, confidence=confidence)
    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
model = pickle.load(open('decision_tree_model.pkl', 'rb'))


@app.route('/')
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = int(request.form['Gender'])
        Total_Bilirubin = float(request.form['Total_Bilirubin'])
        Direct_bilirubin = float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens = float(request.form['Total_Protiens'])
        Albumin = float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])

        v1 = pickle.load(open('minmax.pkl', 'rb'))
        values1 = v1.fit_transform([[Age, Gender, Total_Bilirubin, Direct_bilirubin, Alkaline_Phosphotase,
                                     Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin,
                                     Albumin_and_Globulin_Ratio]])
        # values1 = values1.reshape()

        prediction = model.predict(values1)

        return render_template('result.html', predict=prediction)


if __name__ == "__main__":
    app.run(host='127.73.80.7', port=5080, debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model
model_path = r'C:\Users\hp\Documents\TEXT BOOKS\year 3_sem2\jupyternotebooks\best_rf_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the scaler used during model training
scaler_path = r'C:\Users\hp\Documents\TEXT BOOKS\year 3_sem2\jupyternotebooks\scaler.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Define a route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get user input from the form
        potential = float(request.form['potential'])
        value_eur = float(request.form['value_eur'])
        wage_eur = float(request.form['wage_eur'])
        passing = float(request.form['passing'])
        dribbling = float(request.form['dribbling'])
        attacking_short_passing = float(request.form['attacking_short_passing'])
        movement_reactions = float(request.form['movement_reactions'])
        power_shot_power = float(request.form['power_shot_power'])
        mentality_vision = float(request.form['mentality_vision'])
        mentality_composure = float(request.form['mentality_composure'])
        skill_long_passing= float(request.form['skill_long_passing'])
        physic=float(request.form['physic'])
        age=float(request.form['age'])
        skill_ball_control=float(request.form['skill_ball_control'])
        international_reputation=float(request.form['international_reputation'])

        # Make predictions using the loaded model
        input_data = pd.DataFrame(data=[[potential, value_eur, wage_eur, skill_long_passing, physic,
                                            dribbling, attacking_short_passing, movement_reactions,
                                            power_shot_power,passing,age,skill_ball_control,international_reputation,
                                           mentality_vision,mentality_composure]],
                                  columns=['potential', 'value_eur', 'wage_eur', 'skill_long_passing','physic'
                                           'dribbling', 'attacking_short_passing', 'movement_reactions',
                                           'power_shot_power','passing','age','skill_ball_control','international_reputation',
                                           'mentality_vision', 'mentality_composure'])


        # Calculate prediction intervals using bootstrapping
        num_samples = 5
        predictions = []
        for i in range(num_samples):
            scaled_input = scaler.transform(input_data)
            result = model.predict(scaled_input)
            predictions.append(result[0])

        # The lower and upper bounds represent a range within which you
        # can be confident that the true prediction falls.
        lower_bound = np.percentile(predictions, 2.5)
        upper_bound = np.percentile(predictions, 97.5)


        response = {
            'prediction': float(result[0]),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
        alpha = response['upper_bound'] - response['lower_bound']
        actual_confidence = (1 - alpha)*100


        return redirect(url_for('home', prediction_result=result[0], actual_confidence=actual_confidence))

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)

   
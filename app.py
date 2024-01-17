from flask import Flask, render_template, request, redirect, url_for
from googletrans import Translator
import pickle

app = Flask(__name__)

def load_classifier(model_filename='logistic_regression_classifier.pkl'):
    with open(model_filename, 'rb') as model_file:
        return pickle.load(model_file)

def predict_archetype_with_translation(user_responses):
    loaded_classifier = load_classifier()

    # Translate the combined user responses to English using googletrans
    translator = Translator()
    english_text = translator.translate(user_responses, src='auto', dest='en').text

    # Make predictions for the translated text
    prediction = loaded_classifier.predict([english_text])

    # Get the probabilities for each class
    probabilities = loaded_classifier.predict_proba([english_text])[0]

    # Create a list to store each archetype, its probability, and description
    archetype_results = []
    for archetype, prob in zip(loaded_classifier.classes_, probabilities):
        description = get_archetype_description(archetype)
        archetype_results.append({'archetype': archetype, 'probability': prob, 'description': description})

    return {'predictedGroup': prediction[0], 'archetypeResults': archetype_results}

def get_archetype_description(archetype):
    descriptions = {
        'Visionary': 'Visionairs zijn creatieve en ruimdenkende individuen die zich richten op mogelijkheden en worden gedreven door hun idealen.',
        'Organizer': 'Organisatoren zijn praktisch ingesteld en hebben oog voor detail. Ze blinken uit in plannen, analyseren en het brengen van orde in complexe situaties.',
        'Connector': 'Connectors zijn gericht op mensen en empathisch, bedreven in het opbouwen van sterke relaties en het bieden van ondersteuning',
        'Guide': 'Gidsen zijn analytische en logische leiders die duidelijke richting geven en zich richten op het behalen van praktische resultaten'
    }

    return descriptions.get(archetype, 'Unknown archetype.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_quiz')
def start_quiz():
    return redirect(url_for('quiz_question', question_number=1))

@app.route('/quiz_question/<int:question_number>', methods=['GET', 'POST'])
def quiz_question(question_number):
    if request.method == 'POST':
        user_responses = " ".join([request.form.get(f'question_{i}', '') for i in range(1, question_number + 1)])
        if question_number == 3:
            # Perform prediction
            prediction_result = predict_archetype_with_translation(user_responses)
            # Render predict.html with the prediction result
            return render_template('predict.html', prediction_result=prediction_result)

        return redirect(url_for('quiz_question', question_number=question_number + 1))

    return render_template(f'quiz_question_{question_number}.html')

if __name__ == '__main__':
    app.run(debug=True)

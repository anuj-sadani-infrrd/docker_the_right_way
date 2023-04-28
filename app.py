from flask import Flask, request, jsonify
import spacy

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")
print("Spacy model loaded successfully!")
print("Model Ready for predictions \U0001F680")

# Define the Flask app
app = Flask(__name__)

# Define a route for text classification prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get the text input from the request body
    text = request.json["text"]

    # Process the text with Spacy
    doc = nlp(text)

    # Find named entities, phrases and concepts
    orgs = [entity.text for entity in doc.ents if entity.label_ == "ORG"]

    # Return the prediction result as JSON
    return jsonify({"ORGS": orgs})

if __name__ == "__main__":
    # This will have issue with docker, as it will not be able to access the port and localhost
    # app.run(debug=True)
    app.run(host="0.0.0.0",port=5000, debug=True)
import requests
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# ================= LOAD MODELS =================
crop_model = pickle.load(open('models/crop_model.pkl', 'rb'))
fert_model = pickle.load(open('models/fertilizer_model.pkl', 'rb'))
npk_model = pickle.load(open('models/npk_model.pkl', 'rb'))
disease_model = load_model('models/plant_disease_model.h5')

# ================= CLASS NAMES =================
class_names = os.listdir("dataset/train")
class_names.sort()

# ================= DISEASE SOLUTIONS =================
disease_solutions = {
    "Tomato____Leaf_Mold": [
        "Apply fungicide (Mancozeb or copper-based spray)",
        "Avoid overwatering, especially in evening",
        "Maintain proper spacing between plants",
        "Remove infected leaves immediately",
        "Prefer morning irrigation"
    ],

    "Tomato____Early_blight": [
        "Use Fungicide (a spray that controls fungal infection in plants)",
        "Remove infected leaves",
        "Do not pour water on leaves",
        "Maintain proper spacing",
        "Use healthy seeds"
    ],

    "Tomato____Late_blight": [
        "Apply fungicide immediately",
        "Avoid excess moisture",
        "Use resistant varieties",
        "Ensure proper drainage"
    ],

    "Corn____Common_rust": [
        "Use resistant crop varieties",
        "Apply fungicide if infection is severe",
        "Maintain proper spacing"
    ],

    "Healthy": [
        "Plant is healthy",
        "Maintain proper irrigation and nutrients"
    ]
}


# ================= HOME =================
@app.route('/')
def home():
    return render_template('home.html')

# ================= NPK AUTO GENERATION =================
@app.route('/predict_npk', methods=['POST'])
def predict_npk():
    try:
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        npk = npk_model.predict([[temperature, humidity, ph, rainfall]])
        N, P, K = npk[0]

        return jsonify({
            "N": round(N, 2),
            "P": round(P, 2),
            "K": round(K, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ================= CROP PAGE =================
@app.route('/crop')
def crop():
    return render_template('crop.html')

# ================= SENSOR DATA =================
@app.route('/get_sensor')
def get_sensor():

    try:

        url = "https://api.thingspeak.com/channels/3277509/feeds/last.json?api_key=G31XS9QT0CVGY9LS"

        response = requests.get(url, timeout=10)

        data = response.json()

        print("LATEST:", data)

        return jsonify({

            # field3 = temperature
            "temperature": float(data['field3'] or 33.30),

            # field4 = humidity
            "humidity": float(data['field4'] or 58.0),

            # field2 = rain
            "rainfall": float(data['field2'] or 0),

            # field1 = soil
            "soil": float(data['field1'] or 3547)

        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "temperature": 0,
            "humidity": 0,
            "rainfall": 0,
            "soil": 0
        })
# ================= CROP PREDICTION =================
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_model.predict(data)

        crop = prediction[0]
        
        # 🔥 CHANGE OUTPUT NAME HERE
        
        crop_mapping = {
                        "mothbeans": "Green Chilli",
                        "pigeonpeas": "Toor Dal",
                        "kidneybeans": "Rajma",
                        "chickpea": "Chickpea (Chana)"}
        



        crop_mapping = {
                        "rice": "Rice",
                        "maize": "Maize (Corn)",
                        "chickpea": "Chickpea (Chana)",
                        "kidneybeans": "Rajma",
                        "pigeonpeas": "Toor Dal (Arhar)",
                        "mothbeans": "Green Chilli",
                        "mungbean": "Green Gram (Moong)",
                        "blackgram": "Black Gram (Urad Dal)",
                        "lentil": "Lentil (Masoor Dal)",
    
                        "pomegranate": "Pomegranate",
                        "banana": "Banana",
                        "mango": "Mango",
                        "grapes": "Grapes",
                        "watermelon": "Watermelon",
    
                        "apple": "Apple",
                        "orange": "Orange",
                        "papaya": "Papaya",
                        "coconut": "Coconut",
    
                        "cotton": "Cotton",
                        "jute": "Jute",
                        "coffee": "Coffee",
    
                        "brinjal": "Brinjal (Eggplant)",
                        "green chilli": "Green Chilli",
                        "ladies_finger": "Ladies Finger (Okra)"
}
        
        crop = crop_mapping.get(crop, crop)
        

        return render_template('result.html', result=crop)

    except Exception as e:
        return str(e)

# ================= FERTILIZER =================
@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])

        data = np.array([[N, P, K]])
        prediction = fert_model.predict(data)

        return render_template('result.html', result=prediction[0])

    except Exception as e:
        return str(e)

# ================= DISEASE PAGE =================
@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    try:
        file = request.files['image']

        if not os.path.exists("static"):
            os.makedirs("static")

        path = os.path.join("static", file.filename)
        file.save(path)

        # Process image
        img = image.load_img(path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255

        # Prediction
        pred = disease_model.predict(img_array)
        result = class_names[np.argmax(pred)]

        # Clean name
        clean_name = result.replace("_", " ").replace("_", " ")

        # Get solution
        solution = disease_solutions.get(result, ["No solution available"])

        return render_template('result.html', result=clean_name, solution=solution)

    except Exception as e:
        return str(e)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, port=5001)
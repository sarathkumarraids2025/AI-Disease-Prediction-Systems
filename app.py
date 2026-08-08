from flask import Flask, render_template, request, send_file, redirect
import numpy as np
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import matplotlib.pyplot as plt
import os


# ==========================================
# LOAD SYMPTOMS
# ==========================================

symptom_list = joblib.load("symptom_list.pkl")


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

report_data = {}


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("model.pkl")

label_encoder = joblib.load("label_encoder.pkl")



# ==========================================
# DISEASE INFORMATION
# ==========================================

disease_info = {


    "Arthritis": {

    "description":
    "Arthritis is a condition that causes joint pain and inflammation.",

    "precautions":[
        "Do regular exercise",
        "Avoid excessive joint stress",
        "Maintain healthy weight"
    ],

    "diet":[
        "Omega-3 rich foods",
        "Fruits",
        "Vegetables"
    ],

    "doctor":
    "Orthopedic Doctor"
},
}


# ==========================================
# EMERGENCY SYMPTOMS
# ==========================================

emergency_symptoms = [

    "chest pain",
    "breathlessness",
    "high fever",
    "severe bleeding",
    "unconsciousness"

]
# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")



# ==========================================
# LOGIN
# ==========================================

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username","").strip()

        password = request.form.get("password","").strip()


        if username == "" or password == "":

            return render_template(
                "login.html",
                error="Please enter Username and Password"
            )


        return redirect("/dashboard")


    return render_template("login.html")



# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")



# ==========================================
# ABOUT
# ==========================================

@app.route("/about")
def about():

    return render_template("about.html")



# ==========================================
# CONTACT
# ==========================================

@app.route("/contact")
def contact():

    return render_template("contact.html")



# ==========================================
# PREDICTION PAGE
# ==========================================

@app.route("/prediction")
def prediction():
    return render_template(
        "prediction.html",
        symptoms=symptom_list
    )

    # your existing prediction code
    diseases = [
        "Arthritis",
        "Diabetes",
        "Flu"
    ]

    confidence = [
        80,
        65,
        45
    ]


    plt.figure(figsize=(6,4))

    plt.bar(diseases, confidence)

    plt.xlabel("Disease")
    plt.ylabel("Confidence %")

    plt.title("Top Disease Predictions")


    graph_path = "static/disease_graph.png"

    plt.savefig(graph_path)

    plt.close()


    return render_template(
        "result.html",
        graph="disease_graph.png"
    )

    return render_template("prediction.html")



# ==========================================
# PREDICT DISEASE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():


    global report_data


    name = request.form["name"]

    age = request.form["age"]

    gender = request.form["gender"]



    symptoms = request.form.getlist("symptoms")

    selected_symptoms = symptoms



    # Create Input Vector

    input_data = np.zeros(len(symptom_list))


    for symptom in symptoms:

        if symptom in symptom_list:

            index = symptom_list.index(symptom)

            input_data[index] = 1



    input_data = input_data.reshape(1,-1)



    # Prediction

    prediction = model.predict(input_data)[0]


    disease = label_encoder.inverse_transform(
        [prediction]
    )[0]



    # ==========================================
    # CONFIDENCE
    # ==========================================


    probabilities = model.predict_proba(input_data)[0]


    confidence = round(
        max(probabilities)*100,
        2
    )



    # ==========================================
    # TOP 3
    # ==========================================


    top_indices = probabilities.argsort()[-3:][::-1]


    top3=[]


    for i in top_indices:


        disease_name = label_encoder.inverse_transform([i])[0]


        probability = round(
            probabilities[i]*100,
            2
        )


        top3.append({

            "disease": disease_name,

            "probability": probability

        })
        # ==========================================
    # EMERGENCY ALERT
    # ==========================================

    emergency = False

    emergency_message = ""


    for symptom in symptoms:


        if symptom.lower() in emergency_symptoms:


            emergency = True


            emergency_message = (
                "🚨 HIGH RISK ALERT! "
                "Please visit the nearest hospital immediately."
            )


            break



    # ==========================================
    # DISEASE DETAILS
    # ==========================================


    if disease in disease_info:


        description = disease_info[disease]["description"]

        precautions = disease_info[disease]["precautions"]

        diet = disease_info[disease]["diet"]

        doctor = disease_info[disease]["doctor"]



    else:


        description = "Please consult a doctor."


        precautions = [

            "Take proper rest",

            "Consult doctor"

        ]


        diet = [

            "Healthy food",

            "Drink enough water"

        ]


        doctor = "General Physician"





    # ==========================================
    # SAVE DATA FOR PDF
    # ==========================================


    report_data = {


        "name": name,

        "age": age,

        "gender": gender,

        "disease": disease,

        "confidence": confidence,

        "top3": top3,

        "description": description,

        "precautions": precautions,

        "diet": diet,

        "doctor": doctor

    }





    # ==========================================
    # RESULT PAGE
    # ==========================================


    return render_template(


        "result.html",


        name=name,

        age=age,

        gender=gender,


        prediction=disease,


        confidence=confidence,


        top3=top3,


        description=description,


        precautions=precautions,


        diet=diet,


        doctor=doctor,


        symptoms=selected_symptoms,


        emergency=emergency,


        emergency_message=emergency_message

    )
# ==========================================
# PDF REPORT
# ==========================================

@app.route("/download_report")
def download_report():


    filename = "medical_report.pdf"


    doc = SimpleDocTemplate(filename)


    styles = getSampleStyleSheet()


    content = []



    content.append(

        Paragraph(

            "AI Disease Prediction Medical Report",

            styles["Title"]

        )

    )



    content.append(Spacer(1,20))



    content.append(

        Paragraph(

            f"Generated Date : {datetime.now()}",

            styles["Normal"]

        )

    )



    content.append(Spacer(1,20))



    content.append(

        Paragraph(

            f"Patient Name : {report_data.get('name','N/A')}",

            styles["Normal"]

        )

    )



    content.append(

        Paragraph(

            f"Age : {report_data.get('age','N/A')}",

            styles["Normal"]

        )

    )



    content.append(

        Paragraph(

            f"Gender : {report_data.get('gender','N/A')}",

            styles["Normal"]

        )

    )



    content.append(

        Paragraph(

            f"Predicted Disease : {report_data.get('disease','N/A')}",

            styles["Normal"]

        )

    )



    content.append(

        Paragraph(

            f"Confidence Score : {report_data.get('confidence','N/A')}%",

            styles["Normal"]

        )

    )



    content.append(Spacer(1,20))



    content.append(

        Paragraph(

            "Top 3 Disease Predictions",

            styles["Heading2"]

        )

    )



    for item in report_data.get("top3", []):


        content.append(

            Paragraph(

                f"{item['disease']} - {item['probability']}%",

                styles["Normal"]

            )

        )




    content.append(Spacer(1,20))



    content.append(

        Paragraph(

            "Description",

            styles["Heading2"]

        )

    )


    content.append(

        Paragraph(

            str(report_data.get("description","N/A")),

            styles["Normal"]

        )

    )

    content.append(
    Paragraph(
        "Precautions : "
        + ", ".join(report_data.get("precautions",[])),
        styles["Normal"]
    )
)


    content.append(
    Paragraph(
        "Diet Recommendation : "
        + ", ".join(report_data.get("diet",[])),
        styles["Normal"]
    )
)

    content.append(

        Paragraph(

            "Doctor Recommendation : "

            + str(report_data.get("doctor","N/A")),

            styles["Normal"]

        )

    )



    doc.build(content)



    return send_file(

        filename,

        as_attachment=True

    )



# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
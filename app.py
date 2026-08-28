from flask import Flask, render_template, request
from predict import predict_loan

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = {
            "loan_amnt": float(request.form["loan"]),
            "annual_inc": float(request.form["income"]),
            "int_rate": float(request.form["rate"]),
            "term": int(request.form["term"]),
            "grade": request.form["grade"],
            "dti": float(request.form["dti"])
        }

        prob, decision = predict_loan(data)

        return render_template("index.html", result=decision, prob=prob)

    return render_template("index.html")

if __name__ == "__main__":
    print("Starting Flask")
    app.run(debug=True)
from flask import Flask, render_template, redirect, url_for, request

from backend.test_ import predict

# cors
# from flask_cors import CORS

app = Flask(__name__)
# cors
# CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    search_results = request.form["search"]
    # process the search term and return the search results
    answer = ""
    if search_results == "":
        answer = "Please enter a number"
    else:
        userId = int(search_results)
        if userId < 0 or userId > 31667:
            answer = "Please enter a number between 1 and 31667"
        else:
            answer = predict(userId)
    return render_template("index.html", results=answer)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

from flask import Flask,abort
app=Flask(__name__)

@app.route("/")
def two_hundered():
    return "200 all is good from sepide"
@app.route("/route")
def error():
    abort(500,"oh,some error")

if __name__=="__main__":
    app.run(debug=True,port=8000,host="0.0.0.0")
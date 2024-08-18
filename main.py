import time

from flask import Flask, request, jsonify
import flask_to_flask

app = Flask(__name__)

firstTime = True


@app.route('/set_traffic_lights', methods=['POST'])
def set_traffic_lights():
    global firstTime
    if not firstTime:
        flask_to_flask.getSignalTiming()
    # Assuming you receive some data from the ESP8266
    data = request.json
    print("Data received from ESP:", data)

    print("Data sent to ESP: Road:", flask_to_flask.timeOfEachLane)
    response = jsonify(flask_to_flask.timeOfEachLane)
    firstTime = False
    return response

if __name__ == '__main__':
    flask_to_flask.start()
    #app.run(host='0.0.0.0', port=5000)
    while True:
        time.sleep(60)
        flask_to_flask.getSignalTiming()

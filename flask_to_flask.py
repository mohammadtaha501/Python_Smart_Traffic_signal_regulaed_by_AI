from flask import Flask
from flask_cors import CORS
from time import perf_counter
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import cv2 as cv
from threading import Thread, Lock, Event
import requests

app = Flask(__name__)

cors = CORS(app)

# Define a lock to synchronize access to shared resources
lock = Lock()
# shared resources
thread1list = []
thread2list = []
thread3list = []
thread4list = []

#pause_event = Event()
model = YOLO("yolov8m.pt")

timeOfEachLane = {
    'road1': 0,
    'road2': 0,
    'road3': 0,
    'road4': 0
}

video_stream_url1 = "https://c7.alamy.com/comp/PXY5XN/lone-car-driving-on-a-stretch-of-a130-duel-carriageway-after-exit-ford-kuga-single-vehicle-automobile-on-road-patchwork-road-surfaces-copyspace-PXY5XN.jpg"
video_stream_url2 = "https://c7.alamy.com/comp/PXY5XN/lone-car-driving-on-a-stretch-of-a130-duel-carriageway-after-exit-ford-kuga-single-vehicle-automobile-on-road-patchwork-road-surfaces-copyspace-PXY5XN.jpg"
video_stream_url3 = "https://republicpolicy.com/wp-content/uploads/2023/01/58e1a83697369.jpg"
video_stream_url4 = "https://c7.alamy.com/comp/H06C91/lahore-pakistan-19th-sep-2016-numerous-vehicles-are-stuck-in-traffic-H06C91.jpg"


def model_prediction_function(video_stream_url, mylist, model):
    global lock, thread1list, thread2list, thread3list, thread4list

    x_line = 100
    traffic_vehicles = ["car", "motorcycle", "bus", "truck"]  # Define traffic vehicles

    total_vehicles = 0  # Initialize a counter for total vehicles
    vehicle_counts = {  # Initialize a dictionary for individual vehicle counts
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }

    color = (255, 0, 255)
    # cap = cv.VideoCapture(video_stream_url)
    frame_count = 0
    while True:
        #print("in the loop")
        response = requests.get(video_stream_url)
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv.imdecode(arr, -1)  # 'Load it as it is'
        frame_count += 1
        # ret, img = cap.read()
        # while not ret:
        # print('stream not captured')
        # ret, img = cap.read()
        annotator = Annotator(img)

        stime = perf_counter()
        results = model.predict(img)
        #print('prediction')
        if results is None or len(results) == 0:
            # No objects detected, skip to the next iteration
            continue
        result = results[0]
        #print('step 1')
        for box in result.boxes:
            b = box.xyxy[0]
            if b[1] > x_line:  # Check if object is above the threshold line
                c = box.cls
                #print('step 2')
                class_id = result.names[int(c)]
                if class_id in traffic_vehicles:  # Check if it's a traffic vehicle
                    conf = round(box.conf[0].item(), 2)
                    annotator.box_label(b, f"{class_id} {conf:.2f}", color=color)  # Add label for traffic vehicles only
                    total_vehicles += 1  # Increment total vehicle count
                    vehicle_counts[class_id] += 1  # Increment specific vehicle count
        etime = perf_counter()
        #print('calculating ')
        weightOfLane = vehicle_counts["car"]
        weightOfLane += vehicle_counts["motorcycle"] * 0.5
        weightOfLane += (vehicle_counts["bus"] + vehicle_counts["truck"]) * 2
        mylist.append(weightOfLane)
        #print("total time : ", etime - stime)
        #print(f"Total vehicles: {total_vehicles}")
        #print(f"Vehicle counts: {vehicle_counts}")
        vehicle_counts["car"] = 0
        vehicle_counts["motorcycle"] = 0
        vehicle_counts["bus"] = 0
        vehicle_counts["truck"] = 0
        total_vehicles = 0
        #cv.imshow('detected', img)
        key = cv.waitKey(9000)
        if key == 27:  # if ESC is pressed, exit loop
            cv.destroyAllWindows()
            break


def start():
    global thread1list, thread2list, thread3list, thread4list
    # Create and start threads
    thread1 = Thread(target=model_prediction_function, args=(video_stream_url1, thread1list, model), name="lane 1")
    thread2 = Thread(target=model_prediction_function, args=(video_stream_url2, thread2list, model), name="lane 2")
    thread3 = Thread(target=model_prediction_function, args=(video_stream_url3, thread3list, model), name="lane 3")
    thread4 = Thread(target=model_prediction_function, args=(video_stream_url4, thread4list, model), name="lane 4")

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    #pause_event.set()


def getSignalTiming():
    global lock, thread1list, thread2list, thread3list, thread4list, timeOfEachLane
    # client_socket.connect((server_ip, server_port))
    #pause_event.clear()
    averageWeightOfLAne1 = 0
    averageWeightOfLAne2 = 0
    averageWeightOfLAne3 = 0
    averageWeightOfLAne4 = 0

    print("---------------------------------------------------------------------------------------------")
    with lock:
        if len(thread1list) == 0:
            averageWeightOfLAne1 = 0
        else:
            averageWeightOfLAne1 = sum(thread1list) / len(thread1list)
            print("average weight of lane 4:", averageWeightOfLAne1)
            thread1list.clear()
        if len(thread2list) == 0:
            averageWeightOfLAne2 = 0
        else:
            averageWeightOfLAne2 = sum(thread2list) / len(thread2list)
            print("average weight of lane 2:", averageWeightOfLAne2)
            thread2list.clear()
        if len(thread3list) == 0:
            averageWeightOfLAne3 = 0
        else:
            averageWeightOfLAne3 = sum(thread3list) / len(thread3list)
            print("average weight of lane 3:", averageWeightOfLAne3)
            thread1list.clear()
        if len(thread4list) == 0:
            averageWeightOfLAne4 = 0
        else:
            averageWeightOfLAne4 = sum(thread4list) / len(thread4list)
            print("average weight of lane 4:", averageWeightOfLAne4)
            thread4list.clear()
    # calculating the % of each lane as compare to the sum of the all lane weight
    totalWeight = averageWeightOfLAne1 + averageWeightOfLAne2 + averageWeightOfLAne3 + averageWeightOfLAne4
    if totalWeight == 0:
        totalWeight = 1
    print("sum of all weights:", totalWeight)
    lane1percentage = averageWeightOfLAne1 / totalWeight
    lane2percentage = averageWeightOfLAne2 / totalWeight
    lane3percentage = averageWeightOfLAne3 / totalWeight
    lane4percentage = averageWeightOfLAne4 / totalWeight
    print("road 1 percentage:", lane1percentage, "\n""road 2 percentage:", lane2percentage, "\n""road 3 percentage:",
          lane3percentage, "\n""road 4 percentage:", lane4percentage)
    timeForLane1 = lane1percentage * 61
    timeForLane2 = lane2percentage * 61
    timeForLane3 = lane3percentage * 61
    timeForLane4 = lane4percentage * 61
    # this dictionary will be sent when api is called
    if lane1percentage<0.05 and averageWeightOfLAne1 != 0:
        timeOfEachLane['road1'] = 3
    else:
        timeOfEachLane['road1'] = int(timeForLane1)
    if lane2percentage < 0.05 and averageWeightOfLAne2 != 0:
        timeOfEachLane['road2'] = 3
    else:
        timeOfEachLane['road2'] = int(timeForLane2)
    if lane3percentage<0.05 and averageWeightOfLAne3 != 0:
        timeOfEachLane['road3'] = 3
    else:
        timeOfEachLane['road3'] = int(timeForLane3)
    if lane4percentage<0.05 and averageWeightOfLAne4 != 0:
        timeOfEachLane['road4'] = 3
    else:
        timeOfEachLane['road4'] = int(timeForLane4)
    print("timeOfEachLane : ", timeOfEachLane)
    print("---------------------------------------------------------------------------------------------")
    #pause_event.set()



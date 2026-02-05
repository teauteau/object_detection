import cv2
import numpy as np
import time
import threading
import datetime
from flask import Flask, render_template, render_template_string, jsonify, Response, send_from_directory
from ultralytics import YOLO
import os

STREAM_URL = "https://volkskrant:nrc@192.168.0.107:8080/video" 
MODEL_PATH = "yolov8n.pt"  # lightweight YOLO model
FRAME_WIDTH = 1920  # resize frame width
FRAME_HEIGHT = 1080  # resize frame height



cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

# model = YOLO(MODEL_PATH)
# model.export(format="openvino")
model = YOLO("yolov8n_openvino_model")


class FrameGrabber:
    """Background thread that continuously reads frames and keeps the latest one."""
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # if reading fails, wait briefly and retry
                time.sleep(0.01)
                continue
            with self.lock:
                # store the latest frame
                self.frame = frame

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            # return a copy to avoid race conditions
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


grabber = FrameGrabber(cap)

class_counters = {"person": 0, "bicycle": 0, "car": 0}
class_prev_centers = {"person": {}, "bicycle": {}, "car": {}}
frame_display = None

def video_loop():
    global class_counters, bicycle_count, frame_display, class_prev_centers, frame_display
    try:
        while True:
            print("Fetching latest frame...")
            frame = grabber.get_frame()
            if frame is None:
                # no frame yet from the grabber; wait a bit
                time.sleep(0.1)
                continue

            # Resize frame to reduce processing time
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            cv2.putText(frame, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (10, frame.shape[0] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)


            # Run YOLO detection + tracking
            results = model.track(
                frame,
                persist=True,
                classes=[0, 1, 2, 7],        # 0 = person, 1 = bicycle, 2 = car, 7 = truck
                verbose=False      
            )

            # Map truck (ID 7) to car
            def get_class_name(class_id):
                if class_id == 0:
                    return "person"
                elif class_id == 1:
                    return "bicycle"
                elif class_id == 2 or class_id == 7:
                    return "car"
                else:
                    return None


            if results[0].boxes is not None:
                for box in results[0].boxes:
                    # Skip boxes without tracking ID
                    if box.id is None:
                        continue

                    track_id = int(box.id[0])
                    class_id = int(box.cls[0])
                    class_name = get_class_name(class_id)
                    if class_name is None:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Draw bounding box and center
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

                    # check if object moved from prev frame to current frame
                    if track_id in class_prev_centers[class_name]:
                        if (abs(class_prev_centers[class_name][track_id] - cx) > 10) & (class_prev_centers[class_name][track_id] != -1):
                            class_counters[class_name] += 1
                            print(f"{class_name.capitalize()} ID {track_id} counted")
                            # save image of counted object
                            obj_image = frame.copy()
                            cv2.imwrite(f"objects/{class_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{track_id}.jpg", obj_image)
                            class_prev_centers[class_name][track_id] = -1
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    else:
                        class_prev_centers[class_name][track_id] = cx
                frame_display = cv2.imencode('.jpg', frame)[1].tobytes()

            else:
                 frame_display = cv2.imencode('.jpg', frame)[1].tobytes()

            # Update global counters
            car_count = class_counters["car"]
            person_count = class_counters["person"]
            bicycle_count = class_counters["bicycle"]
            prev_centers = class_prev_centers["car"]
                            
                    

            # Draw counter
            cv2.putText(frame, f"Cars: {car_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"Persons: {person_count}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Bicycles: {bicycle_count}", (20    , 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display
            # cv2.imshow("Car Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # wait 1 second before processing the next most recent frame
            time.sleep(0.3)
    finally:
        grabber.stop()
        cap.release()
        cv2.destroyAllWindows()

threading.Thread(target=video_loop, daemon=True).start()

app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html", count=class_counters["car"], frame=frame_display is not None)

# route to serve the latest frame
@app.route("/frame")
def frame():
    global frame_display
    print("Frame sent to frontend")
    if frame_display is None:
        return Response(status=404)
    return Response(frame_display, mimetype='image/jpeg')

@app.route("/count")
def count():
    return jsonify({"car_count": class_counters["car"], "person_count": class_counters["person"], "bicycle_count": class_counters["bicycle"]})

@app.route("/objects/<filename>")
def objects_image(filename):
    return send_from_directory("objects", filename)

# return a list of all saved car images (latest first)
@app.route("/objects_history_images")
def objects_history_images():
    folder = "objects"
    entries = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        reverse=True
    )
    return jsonify(entries)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

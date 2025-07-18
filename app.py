from flask import Flask, render_template, request, redirect, url_for, send_file, Response, jsonify
import os
import uuid
import shutil
import cv2
import torch
from ultralytics import YOLO
from yolo import run_yolo
from midas_model import run_midas
from ros_launcher import launch_manager, topic_monitor

app = Flask(__name__)

# ===== Folder Setup =====
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
CAPTURE_FOLDER = 'static/captures'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# ===== Camera Initialization =====
def initialize_camera():
    print("[DEBUG] Searching for available camera...")
    for index in range(5):
        cam = cv2.VideoCapture(index, cv2.CAP_V4L2)  # explicitly use V4L2
        if cam.isOpened():
            print(f"[INFO] Camera initialized successfully at index {index}")
            return cam
        cam.release()
    print("[ERROR] No available camera found.")
    return None

camera = initialize_camera()

# ===== Load Models =====
print("[INIT] Loading YOLOv9 model...")
yolo_model = YOLO("yolov9c.pt")

print("[INIT] Loading MiDaS model...")
midas_type = "DPT_Large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas_model = torch.hub.load("intel-isl/MiDaS", midas_type)
midas_model.to(device)
midas_model.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = midas_transforms.dpt_transform if "DPT" in midas_type else midas_transforms.small_transform

# ===== Live Stream Generator =====
def gen_frames():
    if not camera or not camera.isOpened():
        print("[ERROR] gen_frames: Camera not available.")
        return
    while True:
        success, frame = camera.read()
        if not success:
            break
        results = yolo_model.predict(frame, verbose=False)
        annotated = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ===== Routes =====
@app.route('/')
def index():
    return render_template('index.html', camera_error=(camera is None))

@app.route('/video_feed')
def video_feed():
    if not camera or not camera.isOpened():
        return "Camera not available", 503
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    if not camera or not camera.isOpened():
        return "Camera not available", 503
    success, frame = camera.read()
    if success:
        capture_path = os.path.join(CAPTURE_FOLDER, 'capture.jpg')
        cv2.imwrite(capture_path, frame)
        yolo_out = os.path.join(OUTPUT_FOLDER, 'yolo_output.png')
        yolo_mask = os.path.join(OUTPUT_FOLDER, 'yolo_mask.png')
        box_data = os.path.join(OUTPUT_FOLDER, 'bbox_data.csv')
        midas_overlay = os.path.join(OUTPUT_FOLDER, 'depth_overlay.jpg')
        midas_colormap = os.path.join(OUTPUT_FOLDER, 'depth_colormap.jpg')
        run_yolo(capture_path, yolo_out, yolo_mask, box_data, model=yolo_model)
        run_midas(capture_path, midas_colormap, midas_overlay,
                  model=midas_model, transform=midas_transform, device=device)
        return redirect(url_for('results'))
    return "Failed to capture image", 500

@app.route('/results')
def results():
    bbox_csv = ""
    box_data = os.path.join(OUTPUT_FOLDER, 'bbox_data.csv')
    if os.path.exists(box_data):
        with open(box_data, 'r') as f:
            bbox_csv = f.read()
    return render_template('results.html',
                           input_image=url_for('static', filename='captures/capture.jpg'),
                           yolo_output=url_for('static', filename='outputs/yolo_output.png'),
                           mask_output=url_for('static', filename='outputs/yolo_mask.png'),
                           depth_overlay=url_for('static', filename='outputs/depth_overlay.jpg'),
                           depth_colormap=url_for('static', filename='outputs/depth_colormap.jpg'),
                           bbox_table=bbox_csv)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image_file')
    if not file or file.filename == '':
        return redirect(url_for('index'))
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    yolo_out = os.path.join(OUTPUT_FOLDER, 'yolo_output.png')
    yolo_mask = os.path.join(OUTPUT_FOLDER, 'yolo_mask.png')
    box_data = os.path.join(OUTPUT_FOLDER, 'bbox_data.csv')
    midas_overlay = os.path.join(OUTPUT_FOLDER, 'depth_overlay.jpg')
    midas_colormap = os.path.join(OUTPUT_FOLDER, 'depth_colormap.jpg')
    run_yolo(filepath, yolo_out, yolo_mask, box_data, model=yolo_model)
    run_midas(filepath, midas_colormap, midas_overlay,
              model=midas_model, transform=midas_transform, device=device)
    return redirect(url_for('results'))

@app.route('/download')
def download():
    zip_base = os.path.join(OUTPUT_FOLDER, 'results')
    zip_path = shutil.make_archive(zip_base, 'zip', OUTPUT_FOLDER)
    return send_file(zip_path + ".zip", as_attachment=True)

# ===== ROS Launch Controls =====
@app.route('/start_amr')
def start_amr():
    launch_manager.start_amr()
    return redirect(url_for('index'))

@app.route('/stop_amr')
def stop_amr():
    launch_manager.stop_amr()
    return redirect(url_for('index'))

@app.route('/start_mapping')
def start_mapping():
    launch_manager.start_mapping()
    return redirect(url_for('index'))

@app.route('/start_navigation')
def start_navigation():
    launch_manager.start_navigation()
    return redirect(url_for('index'))

@app.route('/topics')
def topics():
    topics = topic_monitor.get_active_topics()
    return jsonify(topics)

# ===== Run App =====
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

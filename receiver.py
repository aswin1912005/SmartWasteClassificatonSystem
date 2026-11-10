from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from datetime import datetime
 
 
import os
import json
import threading
import uuid
import cv2
import torch
from torch import nn
from torchvision import models, transforms


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "MobileNetV3-Small"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best.pt"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "received_images"
)

RESULT_FILE = os.path.join(
    BASE_DIR,
    "latest_result.json"
)

HOST = "10.180.67.79"
PORT = 5000


# =========================================================
# CREATE FOLDER
# =========================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# =========================================================
# LOAD CLASSIFIER
# =========================================================

print("=" * 60)
print("SMART WASTE CLASSIFICATION SYSTEM")
print("=" * 60)

print("Loading MobileNetV3-Small model...")
print("Model:", MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

if checkpoint.get("model_name") != MODEL_NAME:
    raise ValueError(
        f"Expected a {MODEL_NAME} checkpoint, got "
        f"{checkpoint.get('model_name')!r}"
    )

model_classes = checkpoint.get("class_names")
if (
    not isinstance(model_classes, list)
    or not model_classes
    or not all(isinstance(name, str) for name in model_classes)
):
    raise ValueError("Checkpoint must contain a non-empty list of class names")

input_size = checkpoint.get("image_size")
if not isinstance(input_size, int) or input_size <= 0:
    raise ValueError("Checkpoint must contain a positive integer image_size")

model_state_dict = checkpoint.get("model_state_dict")
if not isinstance(model_state_dict, dict):
    raise ValueError("Checkpoint must contain a model_state_dict")

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(model_classes)
)
model.load_state_dict(model_state_dict)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

print("Classifier loaded successfully.")
print("Model classes:")
print(model_classes)

print("=" * 60)


# =========================================================
# GLOBAL STATE
# =========================================================

result_lock = threading.Lock()

# Only ONE image classification job at a time
processing_lock = threading.Lock()

# Current active capture
active_capture_id = None

# Current active sort ID
active_sort_id = None


# =========================================================
# INITIAL RESULT
# =========================================================

latest_result = {

    "status": "waiting",

    "classification": "Waiting",

    "raw_class": "",

    "confidence": 0.0,

    "image": None,

    "timestamp": None,

    "sort_command": "WAIT",

    "capture_id": None,

    "sort_id": None,

    "model": MODEL_NAME,

    "input_size": f"{input_size}x{input_size}"
}


# =========================================================
# SAVE RESULT
# =========================================================

def save_result(data):

    global latest_result

    with result_lock:

        latest_result = data

        try:

            with open(
                RESULT_FILE,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=4
                )

        except Exception as e:

            print(
                "Could not save result:",
                e
            )


# =========================================================
# RESET OLD RESULT ON STARTUP
# =========================================================

startup_result = {

    "status": "waiting",

    "classification": "Waiting",

    "raw_class": "",

    "confidence": 0.0,

    "image": None,

    "timestamp": None,

    "sort_command": "WAIT",

    "capture_id": None,

    "sort_id": None,

    "model": MODEL_NAME,

    "input_size": f"{input_size}x{input_size}"
}

save_result(startup_result)


# =========================================================
# CLASS NORMALIZATION
# =========================================================

def classify_waste(class_name):

    name = str(
        class_name
    ).strip().lower()

    if name in ["bio", "biodegradable"]:

        return (
            "Biodegradable",
            "LEFT"
        )

    if name in [
        "non-bio",
        "non_bio",
        "non-biodegradable",
        "non_biodegradable"
    ]:

        return (
            "Non-biodegradable",
            "RIGHT"
        )

    return (
        "Unknown",
        "UNKNOWN"
    )


# =========================================================
# IMAGE CLASSIFICATION
# =========================================================

def run_classifier(
    image_path,
    filename,
    capture_id,
    sort_id
):

    global active_capture_id
    global active_sort_id

    print()
    print("=" * 60)
    print("CLASSIFICATION STARTED")
    print("=" * 60)

    print("Image:", filename)

    print("Capture ID:", capture_id)

    print("Sort ID:", sort_id)

    try:

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise Exception(
                "Unable to read uploaded image"
            )


        height, width = image.shape[:2]

        print(
            "Image size:",
            width,
            "x",
            height
        )


        # -------------------------------------------------
        # CLASSIFIER
        # -------------------------------------------------

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        input_tensor = preprocess(image_rgb).unsqueeze(0)

        with torch.inference_mode():
            probabilities = torch.softmax(
                model(input_tensor),
                dim=1
            )[0]
            top1_index = int(probabilities.argmax().item())
            confidence = float(probabilities[top1_index].item())

        class_name = model_classes[top1_index]


        classification, sort_command = (
            classify_waste(
                class_name
            )
        )


        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        print(
            "Predicted class:",
            class_name
        )


        print(
            "Classification:",
            classification
        )


        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )


        print(
            "Sort command:",
            sort_command
        )


        print(
            "Capture ID:",
            capture_id
        )


        print(
            "Sort ID:",
            sort_id
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        data = {

            "status": "success",

            "classification":
                classification,

            "raw_class":
                str(class_name),

            "confidence":
                round(
                    confidence * 100,
                    2
                ),

            "image":
                filename,

            "timestamp":
                timestamp,

            "sort_command":
                sort_command,

            "capture_id":
                capture_id,

            "sort_id":
                sort_id,

            "model":
                MODEL_NAME,

            "input_size":
                f"{input_size}x{input_size}"
        }


        save_result(
            data
        )


        print()
        print(
            "Classification completed."
        )


    except Exception as e:

        print()
        print(
            "CLASSIFIER ERROR:",
            str(e)
        )


        data = {

            "status": "error",

            "classification":
                "Error",

            "raw_class":
                "",

            "confidence":
                0,

            "image":
                filename,

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "sort_command":
                "WAIT",

            "capture_id":
                capture_id,

            "sort_id":
                sort_id,

            "error":
                str(e),

            "model":
                MODEL_NAME
        }


        save_result(
            data
        )


    finally:

        # -------------------------------------------------
        # RELEASE PROCESSING LOCK
        # -------------------------------------------------

        processing_lock.release()

        active_capture_id = None

        active_sort_id = None

        print()
        print(
            "CLASSIFICATION PROCESS FINISHED"
        )

        print("=" * 60)
        print()


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    global active_capture_id
    global active_sort_id


    print()
    print("=" * 60)
    print("IMAGE UPLOAD RECEIVED")
    print("=" * 60)


    # -----------------------------------------------------
    # GET CAPTURE ID
    # -----------------------------------------------------

    capture_id = request.form.get(
        "capture_id"
    )


    if not capture_id:

        print(
            "ERROR: No capture_id received"
        )

        return jsonify({

            "status": "error",

            "message":
                "capture_id is required"

        }), 400


    print(
        "Received Capture ID:",
        capture_id
    )


    # -----------------------------------------------------
    # PREVENT SECOND IMAGE BEFORE THE CURRENT SORT CYCLE COMPLETES
    # -----------------------------------------------------

    if not processing_lock.acquire(
        blocking=False
    ):

        print(
            "REJECTED: Previous image is still processing."
        )

        print(
            "Active Capture ID:",
            active_capture_id
        )

        return jsonify({

            "status": "busy",

            "message":
                "Previous image is still being processed.",

            "capture_id":
                active_capture_id,

            "sort_id":
                active_sort_id

        }), 409


    with result_lock:

        previous_status = latest_result.get(
            "status",
            "waiting"
        )

        previous_capture_id = latest_result.get(
            "capture_id"
        )

        previous_sort_id = latest_result.get(
            "sort_id"
        )


    if previous_status not in [
        "waiting",
        "completed",
        "error"
    ]:

        processing_lock.release()

        print(
            "REJECTED: Previous capture has not completed sorting."
        )

        return jsonify({

            "status": "busy",

            "message":
                "Previous capture has not completed sorting.",

            "capture_id":
                previous_capture_id,

            "sort_id":
                previous_sort_id

        }), 409


    if (
        previous_status == "completed"
        and capture_id == previous_capture_id
    ):

        processing_lock.release()

        print(
            "REJECTED: Capture ID has already been processed."
        )

        return jsonify({

            "status": "duplicate",

            "message":
                "This capture has already been processed.",

            "capture_id":
                previous_capture_id

        }), 409


    # -----------------------------------------------------
    # CHECK IMAGE
    # -----------------------------------------------------

    if "image" not in request.files:

        processing_lock.release()

        print(
            "ERROR: No image field"
        )

        return jsonify({

            "status": "error",

            "message":
                "No image received",

            "capture_id":
                capture_id

        }), 400


    file = request.files[
        "image"
    ]


    if file.filename == "":

        processing_lock.release()

        return jsonify({

            "status": "error",

            "message":
                "Empty filename",

            "capture_id":
                capture_id

        }), 400


    # -----------------------------------------------------
    # UNIQUE FILENAME
    # -----------------------------------------------------

    extension = os.path.splitext(
        file.filename
    )[1].lower()


    if extension not in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:

        extension = ".jpg"


    filename = secure_filename(

        datetime.now().strftime(
            "%Y%m%d_%H%M%S_"
        )

        + uuid.uuid4().hex[:8]

        + extension
    )


    filepath = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename
    )


    # -----------------------------------------------------
    # SAVE IMAGE
    # -----------------------------------------------------

    try:

        file.save(
            filepath
        )

    except Exception as e:

        processing_lock.release()

        return jsonify({

            "status": "error",

            "message":
                str(e),

            "capture_id":
                capture_id

        }), 500


    print(
        "Saved image:",
        filepath
    )


    # -----------------------------------------------------
    # NEW SORT ID
    # -----------------------------------------------------

    sort_id = uuid.uuid4().hex[:12]


    active_capture_id = capture_id

    active_sort_id = sort_id


    print(
        "Capture ID:",
        capture_id
    )


    print(
        "New Sort ID:",
        sort_id
    )


    # -----------------------------------------------------
    # PROCESSING STATE
    # -----------------------------------------------------

    processing_data = {

        "status":
            "processing",

        "classification":
            "Processing",

        "raw_class":
            "",

        "confidence":
            0,

        "image":
            filename,

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "sort_command":
            "WAIT",

        "capture_id":
            capture_id,

        "sort_id":
            sort_id,

        "model":
            MODEL_NAME,

        "input_size":
            f"{input_size}x{input_size}"
    }


    save_result(
        processing_data
    )


    # -----------------------------------------------------
    # START CLASSIFIER
    # -----------------------------------------------------

    thread = threading.Thread(

        target=run_classifier,

        args=(

            filepath,

            filename,

            capture_id,

            sort_id
        ),

        daemon=True
    )


    thread.start()


    # -----------------------------------------------------
    # RESPONSE TO ESP32-CAM
    # -----------------------------------------------------

    return jsonify({

        "status":
            "success",

        "message":
            "Image received",

        "filename":
            filename,

        "capture_id":
            capture_id,

        "sort_id":
            sort_id,

        "prediction":
            "processing"

    }), 200


# =========================================================
# LATEST
# =========================================================

@app.route(
    "/latest",
    methods=["GET"]
)
def latest():

    with result_lock:

        return jsonify(
            latest_result
        )


# =========================================================
# SORT RESULT
# =========================================================

@app.route(
    "/sort-result",
    methods=["GET"]
)
def sort_result():

    requested_capture_id = request.args.get(
        "capture_id"
    )


    with result_lock:

        current_capture_id = (
            latest_result.get(
                "capture_id"
            )
        )


        current_status = (
            latest_result.get(
                "status",
                "waiting"
            )
        )


        # -------------------------------------------------
        # IF CONTROLLER PROVIDED CAPTURE ID,
        # ONLY RETURN THAT CAPTURE'S RESULT
        # -------------------------------------------------

        if requested_capture_id:

            if (
                current_capture_id
                != requested_capture_id
            ):

                return jsonify({

                    "status":
                        "waiting",

                    "message":
                        "Waiting for requested capture",

                    "capture_id":
                        current_capture_id,

                    "requested_capture_id":
                        requested_capture_id,

                    "classification":
                        "Waiting",

                    "confidence":
                        0,

                    "sort_command":
                        "WAIT",

                    "sort_id":
                        None

                })


        return jsonify({

            "status":
                current_status,

            "classification":
                latest_result.get(
                    "classification",
                    "Waiting"
                ),

            "confidence":
                latest_result.get(
                    "confidence",
                    0
                ),

            "sort_command":
                latest_result.get(
                    "sort_command",
                    "WAIT"
                ),

            "capture_id":
                current_capture_id,

            "sort_id":
                latest_result.get(
                    "sort_id"
                )

        })


# =========================================================
# SORT COMPLETE
# =========================================================

@app.route(
    "/sort-complete",
    methods=["GET", "POST"]
)
def sort_complete():

    global latest_result


    # -----------------------------------------------------
    # ESP32 SENDS CAPTURE ID
    # -----------------------------------------------------

    requested_capture_id = (
        request.args.get(
            "capture_id"
        )
    )


    with result_lock:

        current_capture_id = (
            latest_result.get(
                "capture_id"
            )
        )


        current_sort_id = (
            latest_result.get(
                "sort_id"
            )
        )


        # -------------------------------------------------
        # REQUIRE CAPTURE ID
        # -------------------------------------------------

        if not requested_capture_id:

            return jsonify({

                "status":
                    "error",

                "message":
                    "capture_id is required"

            }), 400


        # -------------------------------------------------
        # PREVENT WRONG RESULT COMPLETION
        # -------------------------------------------------

        if (
            requested_capture_id
            != current_capture_id
        ):

            return jsonify({

                "status":
                    "error",

                "message":
                    "Capture ID mismatch",

                "current_capture_id":
                    current_capture_id,

                "requested_capture_id":
                    requested_capture_id

            }), 409


        # -------------------------------------------------
        # COMPLETE CURRENT RESULT
        # -------------------------------------------------

        latest_result[
            "sort_command"
        ] = "WAIT"


        latest_result[
            "status"
        ] = "completed"


        latest_result[
            "completed_at"
        ] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        try:

            with open(
                RESULT_FILE,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    latest_result,
                    f,
                    indent=4
                )

        except Exception:
            pass


    print()
    print(
        "SORTING COMPLETED"
    )

    print(
        "Capture ID:",
        requested_capture_id
    )

    print(
        "Sort ID:",
        current_sort_id
    )


    return jsonify({

        "status":
            "success",

        "message":
            "Sorting completed",

        "capture_id":
            requested_capture_id,

        "sort_id":
            current_sort_id

    })


# =========================================================
# IMAGE
# =========================================================

@app.route(
    "/images/<filename>"
)
def image(filename):

    return send_from_directory(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename
    )


# =========================================================
# HEALTH
# =========================================================

@app.route("/health")
def health():

    with result_lock:

        current_status = (
            latest_result.get(
                "status",
                "waiting"
            )
        )


        current_capture = (
            latest_result.get(
                "capture_id"
            )
        )


    return jsonify({

        "status":
            "online",

        "server":
            "Flask",

        "yolo":
            "online",

        "model":
            MODEL_NAME,

        "classes":
            model_classes,

        "device":
            "CPU",

        "input_size":
            f"{input_size}x{input_size}",

        "processing":
            current_status == "processing",

        "capture_id":
            current_capture

    })


# =========================================================
# CLASSIFIER STATUS
# =========================================================

@app.route(
    "/yolo-status"
)
def yolo_status():

    return jsonify({

        "status":
            "online",

        "model":
            MODEL_NAME,

        "model_path":
            MODEL_PATH,

        "classes":
            model_classes,

        "input_size":
            f"{input_size}x{input_size}",

        "device":
            "CPU"

    })


# =========================================================
# CAMERA STATUS
# =========================================================

@app.route(
    "/camera-status"
)
def camera_status():

    return jsonify({

        "status":
            "ESP32-CAM controlled by ESP32",

        "camera_upload":
            "/upload"

    })


# =========================================================
# GEMINI STATUS
# =========================================================

@app.route(
    "/gemini-status"
)
def gemini_status():

    return jsonify({

        "status":
            "disabled",

        "message":
            "Gemini is not used. Local MobileNetV3-Small classification is active."

    })


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SMART WASTE SERVER READY")
    print("=" * 60)

    print()

    print(
        "Website:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    print(
        "LAN Website:"
    )

    print(
        "http://10.180.67.79:5000"
    )

    print()

    print(
        "Upload:"
    )

    print(
        "http://10.180.67.79:5000/upload"
    )

    print()

    print(
        "Latest:"
    )

    print(
        "http://10.180.67.79:5000/latest"
    )

    print()

    print(
        "Sort result:"
    )

    print(
        "http://10.180.67.79:5000/sort-result"
    )

    print()

    print(
        "Health:"
    )

    print(
        "http://10.180.67.79:5000/health"
    )

    print()

    print(
        "Model: MobileNetV3-Small"
    )

    print(
        "Classes:",
        model_classes
    )

    print("=" * 60)

    print()


    app.run(

        host=HOST,

        port=PORT,

        debug=False,

        threaded=True
    )
"""GPU-capable face detection, embedding, matching, and tracking helpers."""

import os
import pickle
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config

PROJECT_ROOT = Path(__file__).resolve().parent


def _configure_cuda_dlls():
    site_packages = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
    if not site_packages.exists():
        import site
        site_packages = Path(site.getsitepackages()[0])
    dll_paths = [
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_runtime" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
        site_packages / "nvidia" / "cufft" / "bin",
        site_packages / "nvidia" / "nvjitlink" / "bin",
    ]
    existing = [str(path) for path in dll_paths if path.exists()]
    if existing:
        os.environ["PATH"] = os.pathsep.join(existing + [os.environ.get("PATH", "")])
        if hasattr(os, "add_dll_directory"):
            for path in existing:
                os.add_dll_directory(path)


_configure_cuda_dlls()
import onnxruntime as ort


MODEL_DIR = PROJECT_ROOT / config.MODEL_DIR
DETECTOR_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
EMBEDDER_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
DETECTOR_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
EMBEDDER_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)


def _download_model(path, url):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(url, path)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _providers():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("CUDAExecutionProvider unavailable; using CPU.")
    return ["CPUExecutionProvider"]


class FaceEngine:
    """Shared YuNet detector and SFace embedder for encoding and live use."""

    def __init__(self):
        _download_model(DETECTOR_PATH, DETECTOR_URL)
        _download_model(EMBEDDER_PATH, EMBEDDER_URL)
        providers = _providers()
        self.detector = ort.InferenceSession(str(DETECTOR_PATH), providers=providers)
        self.embedder = ort.InferenceSession(str(EMBEDDER_PATH), providers=providers)
        active_providers = set(self.detector.get_providers()) & set(self.embedder.get_providers())
        self.provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in active_providers else "CPUExecutionProvider"
        self.detector_input = self.detector.get_inputs()[0]
        self.embedder_input = self.embedder.get_inputs()[0]
        detector_shape = self.detector_input.shape
        self.detector_size = (
            int(detector_shape[-1]),
            int(detector_shape[-2])
        ) if all(isinstance(value, int) for value in detector_shape[-2:]) else (
            config.DETECTOR_SIZE,
            config.DETECTOR_SIZE
        )
        self.cv_detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_PATH),
            "",
            self.detector_size,
            config.DETECTION_CONFIDENCE,
            0.3,
            5000,
        )
        print(f"Face engine: {self.provider}")

    @staticmethod
    def _blob(image, size):
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 128.0,
            size=size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        ).astype(np.float32)

    def detect(self, image, threshold=None):
        threshold = threshold or config.DETECTION_CONFIDENCE
        height, width = image.shape[:2]
        self.cv_detector.setInputSize((width, height))
        _, detections = self.cv_detector.detect(image)
        if detections is None:
            return []
        faces = []
        for detection in detections:
            score = float(detection[14])
            if score < threshold:
                continue
            x, y, box_width, box_height = map(int, detection[:4])
            faces.append({
                "box": (x, y, box_width, box_height),
                "landmarks": detection[4:14].reshape(5, 2),
                "score": score,
            })
        return faces

    def embedding(self, image, face):
        x, y, width, height = face["box"]
        x2 = min(image.shape[1], x + width)
        y2 = min(image.shape[0], y + height)
        crop = image[max(0, y):y2, max(0, x):x2]
        if crop.size == 0:
            return None
        blob = self._blob(crop, (112, 112))
        vector = self.embedder.run(None, {self.embedder_input.name: blob})[0]
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        return vector / (np.linalg.norm(vector) + 1e-8)

    def encode_image(self, image):
        encoded = []
        for face in self.detect(image):
            vector = self.embedding(image, face)
            if vector is not None:
                encoded.append((face, vector))
        return encoded

    def recognize(self, image, known_encodings, known_names):
        results = []
        known = np.asarray(known_encodings, dtype=np.float32)
        for detection, vector in self.encode_image(image):
            label, distance = match_embedding(
                vector,
                known,
                known_names,
                diagnostics=config.FACE_RECOGNITION_DIAGNOSTICS,
                face_box=detection["box"],
            )
            results.append((detection["box"], label, distance))
        return results


class AttributeEngine:
    """Optional ONNX age, gender, and emotion models.

    The model files are intentionally not downloaded or guessed. Each model
    must accept NCHW BGR input and return logits or probabilities. Missing or
    incompatible models produce N/A for that attribute only.
    """

    EMOTIONS = (
        "Neutral",
        "Happy",
        "Surprise",
        "Sad",
        "Angry",
        "Disgust",
        "Fear",
        "Contempt",
    )
    GENDERS = ("Female", "Male")

    def __init__(self):
        self.models = {}
        self.providers = {}
        self.errors = {}
        paths = {
            "age": (config.AGE_MODEL_PATH, config.AGE_DETECTION_ENABLED),
            "gender": (config.GENDER_MODEL_PATH, config.GENDER_DETECTION_ENABLED),
            "emotion": (config.EMOTION_MODEL_PATH, config.EMOTION_DETECTION_ENABLED),
        }
        for name, (path, enabled) in paths.items():
            if not enabled:
                continue
            print(f"Loading {name.title()} model: {path}")
            try:
                if not os.path.exists(path):
                    message = f"file not found: {path}"
                    self.errors[name] = message
                    print(f"ERROR loading {name} model: {message}")
                    continue
                session = ort.InferenceSession(path, providers=_providers())
                self.models[name] = session
                self.providers[name] = (
                    "CUDAExecutionProvider"
                    if "CUDAExecutionProvider" in session.get_providers()
                    else "CPUExecutionProvider"
                )
                print(f"{name.title()} model loaded successfully.")
                print(f"{name.title()} provider: {self.providers[name]}")
            except Exception as error:
                self.errors[name] = str(error)
                print(f"ERROR loading {name} model: {error}")

    @property
    def gpu_enabled(self):
        return bool(self.models) and all(
            provider == "CUDAExecutionProvider" for provider in self.providers.values()
        )

    @staticmethod
    def _input(session, crop, name):
        input_info = session.get_inputs()[0]
        shape = input_info.shape
        channels = int(shape[-3]) if isinstance(shape[-3], int) else 3
        height = int(shape[-2]) if isinstance(shape[-2], int) else 224
        width = int(shape[-1]) if isinstance(shape[-1], int) else 224
        if name == "emotion" and channels == 1:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        if channels == 1:
            blob = resized.astype(np.float32)[None, None, :, :]
        else:
            blob = cv2.dnn.blobFromImage(
                resized,
                scalefactor=1.0 / 127.5,
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
            ).astype(np.float32)
        return input_info.name, blob

    def predict(self, image, box):
        x, y, width, height = box
        crop = image[max(0, y):max(0, y) + height, max(0, x):max(0, x) + width]
        result = {"age": "N/A", "gender": "N/A", "emotion": "N/A"}
        if crop.size == 0:
            return result
        for name, session in self.models.items():
            try:
                input_name, blob = self._input(session, crop, name)
                output = np.asarray(session.run(None, {input_name: blob})[0]).reshape(-1)
                if name == "age":
                    if output.size == 3:
                        result["age"] = str(round(float(output[2]) * 100))
                    elif output.size == 1:
                        value = float(output[0])
                        result["age"] = str(round(value * 100 if 0 <= value <= 1 else value))
                    else:
                        bins = np.arange(output.size, dtype=np.float32)
                        probabilities = np.exp(output - np.max(output))
                        probabilities /= np.sum(probabilities)
                        result["age"] = str(round(float(np.dot(bins, probabilities))))
                elif name == "gender" and output.size >= 2:
                    result["gender"] = self.GENDERS[int(np.argmax(output[:2]))]
                elif name == "emotion" and output.size >= len(self.EMOTIONS):
                    result["emotion"] = self.EMOTIONS[int(np.argmax(output[:8]))]
            except Exception as error:
                print(f"ERROR during {name} inference: {error}")
        return result


_engine = None
_attribute_engine = None


def get_face_engine():
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine


def get_attribute_engine():
    global _attribute_engine
    if _attribute_engine is None:
        _attribute_engine = VerifiedAttributeEngine()
    return _attribute_engine


def recognize_faces_in_image(image, known_encodings, known_names, diagnostics=False):
    results = get_face_engine().recognize(image, known_encodings, known_names)
    return (
        [item[0] for item in results],
        [item[1] for item in results],
        [item[2] for item in results],
    )


def recognize_faces_with_attributes(image, known_encodings, known_names, diagnostics=False):
    face_engine = get_face_engine()
    attribute_engine = get_attribute_engine()
    results = []
    for detection, vector in face_engine.encode_image(image):
        known = np.asarray(known_encodings, dtype=np.float32)
        label, distance = match_embedding(
            vector,
            known,
            known_names,
            diagnostics=diagnostics or config.FACE_RECOGNITION_DIAGNOSTICS,
            face_box=detection["box"],
        )
        attributes = attribute_engine.predict(image, detection["box"])
        results.append((detection["box"], label, distance, attributes))
    return results


def match_embedding(vector, known_encodings, known_names, diagnostics=False, face_box=None):
    """Return a name only when the closest valid encoding passes the threshold."""
    if vector.shape != (128,):
        raise ValueError(f"Expected query embedding shape (128,), got {vector.shape}")
    if len(known_encodings) == 0:
        return "Unknown", 1.0
    if len(known_encodings) != len(known_names):
        raise ValueError(
            f"Encoding/name count mismatch: {len(known_encodings)} vs {len(known_names)}"
        )
    if any(np.asarray(encoding).shape != (128,) for encoding in known_encodings):
        raise ValueError("Every known encoding must have shape (128,)")

    distances = 1.0 - np.dot(known_encodings, vector)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    best_name = known_names[best_index]
    width, height = face_box[2:] if face_box else (0, 0)
    quality_ok = (
        width >= config.MIN_RECOGNITION_FACE_WIDTH
        and height >= config.MIN_RECOGNITION_FACE_HEIGHT
    )
    matched = quality_ok and best_distance <= config.FACE_MATCH_THRESHOLD

    if diagnostics:
        print("\n[FACE MATCH DEBUG]")
        grouped = defaultdict(list)
        for name, distance in zip(known_names, distances):
            grouped[name].append(float(distance))
        for name in sorted(grouped):
            print(f"{name}: {min(grouped[name]):.4f} (best of {len(grouped[name])})")
        print("\nAll stored encodings:")
        for index, (name, distance) in enumerate(zip(known_names, distances)):
            print(f"  [{index:02d}] {name}: {float(distance):.4f}")
        print(f"Minimum:\n{best_name} -> {best_distance:.4f}")
        print(
            f"Face box: {face_box}; threshold: {config.FACE_MATCH_THRESHOLD}; "
            f"minimum size: {config.MIN_RECOGNITION_FACE_WIDTH}x{config.MIN_RECOGNITION_FACE_HEIGHT}"
        )
        print(f"Result: {'MATCH' if matched else 'UNKNOWN'}")

    return (best_name if matched else "Unknown"), best_distance


def audit_encoding_database():
    """Print source-folder and embedding-separation diagnostics without changing files."""
    encodings, names = load_encoded_faces()
    print("\n====================================")
    print("ENCODING DATABASE AUDIT")
    print("====================================")
    print(f"Database path: {Path(config.ENCODINGS_FILE).resolve()}")
    print(f"Known encodings: {len(encodings)}")
    print(f"Known names: {names}")
    print(f"Encoding dimensions: {sorted({np.asarray(item).shape for item in encodings})}")
    print(f"Encoding/name lengths match: {len(encodings) == len(names)}")

    source_files = get_face_files_from_directory(config.KNOWN_FACES_DIR)
    for person in sorted(source_files):
        source_count = len(source_files[person])
        database_count = sum(name == person for name in names)
        print(f"Person: {person}")
        print(f"  Images used: {source_count}")
        print(f"  Encodings: {database_count}")
        if source_count != database_count:
            print("  WARNING: source image and database counts differ")

    vectors = np.asarray(encodings, dtype=np.float32)
    if len(vectors) < 2:
        return
    print("\nCross-person separation:")
    people = sorted(set(names))
    for index, first in enumerate(people):
        first_indices = [i for i, name in enumerate(names) if name == first]
        for second in people[index + 1:]:
            second_indices = [i for i, name in enumerate(names) if name == second]
            distances = 1.0 - np.dot(vectors[first_indices], vectors[second_indices].T)
            print(
                f"  {first} <-> {second}: "
                f"min={float(np.min(distances)):.4f}, "
                f"avg={float(np.mean(distances)):.4f}, "
                f"max={float(np.max(distances)):.4f}"
            )

    duplicate_pairs = []
    for first in range(len(vectors)):
        for second in range(first + 1, len(vectors)):
            if np.allclose(vectors[first], vectors[second], atol=1e-6):
                duplicate_pairs.append((first, second, names[first], names[second]))
    print(f"Exact/near-duplicate encoding pairs: {duplicate_pairs or 'none'}")


class VerifiedAttributeEngine:
    """Open Model Zoo age/gender plus ONNX FER+ emotion inference."""

    EMOTIONS = (
        "Neutral", "Happy", "Surprise", "Sad",
        "Angry", "Disgust", "Fear", "Contempt",
    )
    GENDERS = ("Female", "Male")

    def __init__(self):
        self.errors = {}
        self.providers = {}
        self.age_gender = None
        self.emotion = None
        self._load_age_gender()
        self._load_emotion()

    def _load_age_gender(self):
        model_path = _project_path(config.AGE_MODEL_PATH)
        print(f"Loading Age/Gender model: {model_path}")
        try:
            import openvino as ov

            if not model_path.exists():
                raise FileNotFoundError(model_path)
            core = ov.Core()
            devices = core.available_devices
            device = "GPU" if "GPU" in devices else "CPU"
            model = core.read_model(model_path)
            self.age_gender = core.compile_model(model, device)
            self.providers["age_gender"] = f"OpenVINO:{device}"
            print(f"Age/Gender model loaded successfully.")
            print(f"Age/Gender provider/device: OpenVINO:{device}")
        except Exception as error:
            self.errors["age_gender"] = str(error)
            print(f"ERROR loading Age/Gender model: {model_path}")
            print(error)

    def _load_emotion(self):
        model_path = _project_path(config.EMOTION_MODEL_PATH)
        print(f"Loading Emotion model: {model_path}")
        try:
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            self.emotion = ort.InferenceSession(
                str(model_path),
                providers=_providers(),
            )
            provider = (
                "CUDAExecutionProvider"
                if "CUDAExecutionProvider" in self.emotion.get_providers()
                else "CPUExecutionProvider"
            )
            self.providers["emotion"] = provider
            print("Emotion model loaded successfully.")
            print(f"Emotion provider: {provider}")
        except Exception as error:
            self.errors["emotion"] = str(error)
            print(f"ERROR loading Emotion model: {model_path}")
            print(error)

    @property
    def gpu_enabled(self):
        active = list(self.providers.values())
        return bool(active) and all(
            value in ("CUDAExecutionProvider", "OpenVINO:GPU")
            for value in active
        )

    @staticmethod
    def _crop(image, box, padding=0.15):
        x, y, width, height = box
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image.shape[1], x + width + pad_x)
        bottom = min(image.shape[0], y + height + pad_y)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise ValueError(f"empty face crop for box {box}")
        return crop

    def _predict_age_gender(self, crop):
        input_layer = self.age_gender.input(0)
        input_shape = list(input_layer.shape)
        height, width = int(input_shape[2]), int(input_shape[3])
        resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        blob = resized.transpose(2, 0, 1)[None].astype(np.float32)
        outputs = self.age_gender({input_layer.any_name: blob})
        age_output = None
        gender_output = None
        for output in self.age_gender.outputs:
            name = output.any_name
            values = np.asarray(outputs[output])
            if "fc3_a" in name or values.size == 1:
                age_output = float(values.reshape(-1)[0])
            elif "prob" in name or values.size == 2:
                gender_output = values.reshape(-1)
        if age_output is None or gender_output is None:
            raise ValueError("age/gender outputs fc3_a and prob were not found")
        gender_index = int(np.argmax(gender_output[:2]))
        return {
            "age": str(round(age_output * 100)),
            "gender": self.GENDERS[gender_index],
            "gender_confidence": f"{float(gender_output[gender_index]) * 100:.0f}%",
        }

    def _predict_emotion(self, crop):
        input_layer = self.emotion.get_inputs()[0]
        shape = input_layer.shape
        height, width = int(shape[-2]), int(shape[-1])
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
        blob = gray.astype(np.float32)[None, None, :, :]
        output = np.asarray(self.emotion.run(None, {input_layer.name: blob})[0]).reshape(-1)
        if output.size < len(self.EMOTIONS):
            raise ValueError(f"expected 8 emotion outputs, got {output.size}")
        probabilities = np.exp(output[:8] - np.max(output[:8]))
        probabilities /= np.sum(probabilities)
        index = int(np.argmax(probabilities))
        return {
            "emotion": self.EMOTIONS[index],
            "emotion_index": index,
            "emotion_confidence": f"{float(probabilities[index]) * 100:.0f}%",
            "emotion_probabilities": {
                emotion: float(probabilities[position])
                for position, emotion in enumerate(self.EMOTIONS)
            },
        }

    def predict(self, image, box):
        result = {
            "age": "Unavailable",
            "gender": "Unavailable",
            "emotion": "Unavailable",
        }
        try:
            crop = self._crop(image, box)
        except Exception as error:
            print(f"ERROR creating attribute face crop: {error}")
            return result
        if self.age_gender is not None:
            try:
                result.update(self._predict_age_gender(crop))
            except Exception as error:
                print(f"ERROR during age/gender inference: {error}")
        if self.emotion is not None:
            try:
                result.update(self._predict_emotion(crop))
            except Exception as error:
                print(f"ERROR during emotion inference: {error}")
        return result


def create_trackers(results, frame):
    tracker_factory = getattr(cv2, "legacy", cv2)
    if not hasattr(tracker_factory, "TrackerKCF_create"):
        return []
    trackers = []
    for result in results:
        box, label, distance = result[:3]
        attributes = result[3] if len(result) > 3 else {}
        tracker = tracker_factory.TrackerKCF_create()
        tracker.init(frame, tuple(map(float, box)))
        trackers.append({"tracker": tracker, "label": label, "distance": distance, "attributes": attributes})
    return trackers


def update_trackers(trackers, frame):
    results = []
    active = []
    for item in trackers:
        ok, box = item["tracker"].update(frame)
        if not ok:
            continue
        x, y, width, height = map(int, box)
        if width <= 0 or height <= 0:
            continue
        results.append(((x, y, width, height), item["label"], item["distance"], item["attributes"]))
        active.append(item)
    trackers[:] = active
    return results


def load_encoded_faces():
    if not os.path.exists(config.ENCODINGS_FILE):
        print(f"Warning: Encodings file not found at {config.ENCODINGS_FILE}")
        return [], []
    try:
        with open(config.ENCODINGS_FILE, "rb") as file:
            data = pickle.load(file)
        encodings = data.get("encodings", [])
        names = data.get("names", [])
        if len(encodings) != len(names):
            raise ValueError(
                f"encoding/name count mismatch: {len(encodings)} vs {len(names)}"
            )
        invalid_shapes = sorted({np.asarray(encoding).shape for encoding in encodings if np.asarray(encoding).shape != (128,)})
        if invalid_shapes:
            raise ValueError(f"known encodings must be 128-dimensional; invalid shapes: {invalid_shapes}")
        return encodings, names
    except Exception as error:
        print(f"Error loading encodings: {error}")
        return [], []


def save_encoded_faces(encodings, names):
    os.makedirs(config.ENCODINGS_DIR, exist_ok=True)
    with open(config.ENCODINGS_FILE, "wb") as file:
        pickle.dump({"encodings": encodings, "names": names}, file)
    print(f"Saved {len(encodings)} face encodings to {config.ENCODINGS_FILE}")


def draw_face_boxes(image, face_locations, face_labels, show_distance=False, face_distances=None, face_attributes=None):
    output = image.copy()
    for index, ((x, y, width, height), label) in enumerate(zip(face_locations, face_labels)):
        cv2.rectangle(output, (x, y), (x + width, y + height), config.BOX_COLOR, 2)
        attributes = face_attributes[index] if face_attributes else {}
        lines = [label]
        if config.SHOW_AGE:
            lines.append(f"Age: {attributes.get('age', 'N/A')}")
        if config.SHOW_GENDER:
            gender_confidence = attributes.get("gender_confidence")
            suffix = f" ({gender_confidence})" if gender_confidence else ""
            lines.append(f"Gender: {attributes.get('gender', 'N/A')}{suffix}")
        if config.SHOW_EMOTION:
            emotion_confidence = attributes.get("emotion_confidence")
            suffix = f" ({emotion_confidence})" if emotion_confidence else ""
            lines.append(f"Emotion: {attributes.get('emotion', 'N/A')}{suffix}")
        line_height = 22
        padding = 8
        sizes = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, config.LABEL_FONT_SIZE, config.LABEL_THICKNESS)[0]
            for line in lines
        ]
        panel_width = max(size[0] for size in sizes) + padding * 2
        panel_height = line_height * len(lines) + padding * 2
        panel_x = max(0, min(x, output.shape[1] - panel_width))
        panel_y = y - panel_height - 6
        if panel_y < 0:
            panel_y = min(y + height + 6, output.shape[0] - panel_height)
        panel_y = max(0, panel_y)
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (20, 20, 20),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.82, output, 0.18, 0, output)
        cv2.rectangle(
            output,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            config.BOX_COLOR,
            1,
        )
        for line_index, text in enumerate(lines):
            baseline = panel_y + padding + (line_index + 1) * line_height - 5
            cv2.putText(
                output,
                text,
                (panel_x + padding, baseline),
                cv2.FONT_HERSHEY_DUPLEX,
                config.LABEL_FONT_SIZE,
                (255, 255, 255),
                config.LABEL_THICKNESS,
            )
    return output


def get_face_files_from_directory(directory):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    result = {}
    if not os.path.exists(directory):
        return result
    for person in os.listdir(directory):
        person_path = os.path.join(directory, person)
        if os.path.isdir(person_path):
            result[person] = [
                os.path.join(person_path, filename)
                for filename in os.listdir(person_path)
                if Path(filename).suffix.lower() in valid_extensions
            ]
    return result


def load_image(image_path):
    return cv2.imread(image_path) if os.path.exists(image_path) else None


def normalize_image_path(image_path):
    """Normalize a pasted Windows image path and return it when it is a file."""
    if image_path is None:
        return None

    value = os.path.expandvars(str(image_path).strip())
    if value.startswith("&"):
        value = value[1:].strip()

    if len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0]:
        value = value[1:-1].strip()

    while value.endswith(("'", '"')):
        value = value[:-1].rstrip()

    path = Path(value).expanduser().resolve(strict=False)
    print(f"[IMAGE] Normalized path: {path}")
    if not path.exists() or not path.is_file():
        return None
    return str(path)


def save_image(image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return cv2.imwrite(output_path, image)

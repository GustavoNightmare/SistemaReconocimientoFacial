import os
import time
import threading

import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify
from deepface import DeepFace

app = Flask(__name__)

# ================== CONFIG ==================

DATASET_DIR = "dataset"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Modelo de DeepFace y métrica
DF_MODEL_NAME = "Facenet512"
DF_THRESH = 0.30  # umbral para distancia coseno (más bajo = más estricto)

# Base de embeddings: lista de dicts {id, name, embedding}
person_db = []
db_lock = threading.Lock()

# ================== CÁMARA GLOBAL ==================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara en el índice 0")
else:
    print("✅ Cámara abierta en el índice 0")

cap_lock = threading.Lock()


def get_frame():
    """Lee un frame de la cámara de forma segura."""
    with cap_lock:
        if not cap.isOpened():
            print("⚠ cap no está abierto")
            return None
        ret, frame = cap.read()
    if not ret:
        print("⚠ cap.read() devolvió False (no frame)")
        return None
    return frame


# ================== CLASIFICADOR DE ROSTRO ==================

if not os.path.exists(CASCADE_PATH):
    print(f"ADVERTENCIA: no se encontró {CASCADE_PATH}. "
          f"Descárgalo de OpenCV y colócalo junto a app_deepface.py.")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ================== DATASET & EMBEDDINGS ==================


def ensure_dataset_dir():
    os.makedirs(DATASET_DIR, exist_ok=True)


def get_or_create_person_dir(nombre):
    """
    Crea (o reutiliza) una carpeta tipo: 1_Nombre, 2_OtraPersona, etc.
    """
    ensure_dataset_dir()
    max_id = 0
    existing_id = None

    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        try:
            id_str, name_str = folder.split("_", 1)
            pid = int(id_str)
            max_id = max(max_id, pid)
            if name_str == nombre:
                existing_id = pid
        except ValueError:
            continue

    if existing_id is not None:
        person_id = existing_id
    else:
        person_id = max_id + 1

    folder_name = f"{person_id}_{nombre}"
    folder_path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return person_id, folder_path


def cosine_distance(a, b):
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def build_embeddings():
    """
    Recorre dataset/ y genera un embedding medio por persona usando DeepFace.
    """
    global person_db

    ensure_dataset_dir()
    new_db = []

    print("🔄 Construyendo base de embeddings DeepFace...")

    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        try:
            id_str, name_str = folder.split("_", 1)
            pid = int(id_str)
        except ValueError:
            continue

        embeddings_this = []

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # DeepFace acepta arrays; usamos detector 'skip' porque ya está recortado
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                reps = DeepFace.represent(
                    img_path=img_rgb,
                    model_name=DF_MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False
                )
            except Exception as e:
                print(f"⚠ Error represent en {img_path}: {e}")
                continue

            if not reps:
                continue

            emb = np.array(reps[0]["embedding"], dtype="float32")
            embeddings_this.append(emb)

        if embeddings_this:
            mean_emb = np.mean(embeddings_this, axis=0)
            new_db.append({
                "id": pid,
                "name": name_str,
                "embedding": mean_emb
            })
            print(f"  ➜ {name_str}: {len(embeddings_this)} imágenes")

    with db_lock:
        person_db = new_db

    print(f"✅ Base DeepFace creada con {len(person_db)} personas.")


def deepface_match(embedding):
    """
    Devuelve (nombre_mejor, distancia_mejor) o (None, None) si no hay base.
    """
    with db_lock:
        db_copy = list(person_db)

    if not db_copy:
        return None, None

    best_name = None
    best_dist = 1e9

    for person in db_copy:
        dist = cosine_distance(embedding, person["embedding"])
        if dist < best_dist:
            best_dist = dist
            best_name = person["name"]

    return best_name, best_dist


# ================== VIDEO STREAMING ==================

def gen_stream():
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        with db_lock:
            has_db = len(person_db) > 0

        for (x, y, w, h) in faces:
            face_color = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB)

            color = (0, 255, 255)  # amarillo: detectado
            text = "Rostro detectado"

            if has_db:
                try:
                    reps = DeepFace.represent(
                        img_path=face_rgb,
                        model_name=DF_MODEL_NAME,
                        detector_backend="skip",
                        enforce_detection=False
                    )
                    if reps:
                        emb = np.array(reps[0]["embedding"], dtype="float32")
                        name, dist = deepface_match(emb)
                        if name is not None and dist < DF_THRESH:
                            color = (0, 255, 0)  # verde: reconocido
                            text = f"{name} ({dist:.2f})"
                        else:
                            color = (0, 0, 255)  # rojo: desconocido
                            text = f"Desconocido ({dist:.2f})"
                except Exception as e:
                    # Si algo falla, lo dejamos como detectado
                    print(f"⚠ Error en DeepFace.represent stream: {e}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 25), (x + w, y), color, cv2.FILLED)
            cv2.putText(frame, text, (x + 5, y - 7),
                        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(gen_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ================== PÁGINA PRINCIPAL ==================

@app.route("/")
def index():
    mensaje = request.args.get("mensaje", "")
    return render_template("index.html", mensaje=mensaje)


# ================== REGISTRAR ROSTRO ==================

@app.route("/registrar", methods=["POST"])
def registrar():
    nombre = request.form.get("nombre") or f"persona_{int(time.time())}"

    if face_cascade.empty():
        mensaje = ("El clasificador de rostros no se cargó. "
                   "Revisa la ruta de haarcascade_frontalface_default.xml")
        return render_template("index.html", mensaje=mensaje)

    person_id, person_dir = get_or_create_person_dir(nombre)
    capturadas = 0
    intentos = 0
    max_capturas = 20
    max_intentos = 80

    while capturadas < max_capturas and intentos < max_intentos:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            intentos += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        if len(faces) == 0:
            intentos += 1
            time.sleep(0.1)
            continue

        (x, y, w, h) = faces[0]
        face_color = frame[y:y+h, x:x+w]

        # Redimensionamos a algo razonable; DeepFace lo volverá a ajustar
        face_color = cv2.resize(face_color, (224, 224))

        filename = f"{int(time.time())}_{capturadas}.jpg"
        path = os.path.join(person_dir, filename)
        cv2.imwrite(path, face_color)
        capturadas += 1
        intentos += 1
        time.sleep(0.1)

    if capturadas == 0:
        mensaje = "No se pudo capturar ningún rostro. Intenta acercarte más a la cámara."
    else:
        mensaje = f"Se guardaron {capturadas} imágenes para: {nombre}"
        # regenerar base de embeddings
        build_embeddings()

    return render_template("index.html", mensaje=mensaje)


# ================== RECONOCER (BOTÓN / JAVA) ==================

def reconocer_desde_camara():
    if face_cascade.empty():
        return False, None, None, "no_cascade"

    with db_lock:
        has_db = len(person_db) > 0

    if not has_db:
        return False, None, None, "no_data"

    frame = get_frame()
    if frame is None:
        return False, None, None, "no_camera"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )
    if len(rects) == 0:
        return False, None, None, "no_face"

    (x, y, w, h) = rects[0]
    face_color = frame[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB)

    try:
        reps = DeepFace.represent(
            img_path=face_rgb,
            model_name=DF_MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False
        )
    except Exception as e:
        print(f"⚠ Error en DeepFace.represent reconocer: {e}")
        return False, None, None, "no_embedding"

    if not reps:
        return False, None, None, "no_embedding"

    emb = np.array(reps[0]["embedding"], dtype="float32")
    name, dist = deepface_match(emb)

    if name is not None and dist < DF_THRESH:
        return True, name, dist, None
    else:
        return False, name, dist, None


@app.route("/reconocer", methods=["POST"])
def reconocer():
    autorizado, nombre, dist, error = reconocer_desde_camara()

    if error == "no_cascade":
        mensaje = ("El clasificador de rostros no se cargó. "
                   "Revisa la ruta de haarcascade_frontalface_default.xml")
    elif error == "no_data":
        mensaje = "No hay rostros guardados aún."
    elif error == "no_camera":
        mensaje = "No se pudo leer la cámara."
    elif error == "no_face":
        mensaje = "No se detectó ningún rostro. Coloca tu cara frente a la cámara."
    elif error == "no_embedding":
        mensaje = "No se pudo generar embedding del rostro."
    else:
        if autorizado:
            mensaje = f"✅ Puede entrar: {nombre} (distancia {dist:.2f})"
        else:
            if nombre is None:
                mensaje = "⛔ No puede entrar (desconocido)"
            else:
                mensaje = f"⛔ No puede entrar: {nombre} (distancia {dist:.2f} >= umbral)"

    return render_template("index.html", mensaje=mensaje)


@app.route("/validar-rostro", methods=["POST"])
def validar_rostro():
    """
    Endpoint pensado para que lo llame tu aplicación Java.
    Devuelve JSON con authorized/nombre/distancia/error
    """
    autorizado, nombre, dist, error = reconocer_desde_camara()
    return jsonify({
        "authorized": bool(autorizado),
        "nombre": nombre,
        "distance": dist,
        "error": error
    })


# ================== MAIN ==================

if __name__ == "__main__":
    # construimos base de embeddings si ya hay dataset
    build_embeddings()
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        with cap_lock:
            if cap.isOpened():
                cap.release()

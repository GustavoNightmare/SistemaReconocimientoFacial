import os
import time
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# Carpeta donde se guardan los embeddings de las personas
DB_DIR = "deepface_db"
os.makedirs(DB_DIR, exist_ok=True)

# Cada cuántos frames hacer reconocimiento (30 ≈ 1s si tu cam va a ~30 FPS)
RECOGNITION_INTERVAL = 30

# ---------------------------------------------------------
#  Embedding con DeepFace
# ---------------------------------------------------------


def embed_face(face_bgr):
    """
    Devuelve el embedding de un rostro como np.array(float32).
    Intenta usar Facenet512; si la versión de DeepFace no lo soporta,
    cae al modelo por defecto.
    """
    try:
        # Intentar con Facenet512 (firma nueva)
        try:
            reps = DeepFace.represent(
                face_bgr,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="opencv",
            )
        except TypeError:
            # Versión de DeepFace sin 'model_name' o firma distinta
            reps = DeepFace.represent(
                face_bgr,
                enforce_detection=False,
                detector_backend="opencv",
            )

        if not reps:
            return None

        rep = reps[0]

        # DeepFace a veces devuelve lista de dicts, a veces directamente el vector
        if isinstance(rep, dict) and "embedding" in rep:
            emb = np.array(rep["embedding"], dtype="float32")
        else:
            emb = np.array(rep, dtype="float32")

        if emb.ndim != 1:
            emb = emb.flatten()

        return emb

    except Exception as e:
        print(f"   ⚠️ Error en embed_face / DeepFace.represent: {e}")
        return None

# ---------------------------------------------------------
#  Guardar / cargar embeddings en disco
# ---------------------------------------------------------


def save_embedding(name, emb):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = f"{name}__{ts}.npy"
    path = os.path.join(DB_DIR, fn)
    np.save(path, emb)


def load_db_embeddings():
    db = []
    if not os.path.isdir(DB_DIR):
        return db

    for fn in os.listdir(DB_DIR):
        if not fn.lower().endswith(".npy"):
            continue

        name = fn.split("__", 1)[0]
        path = os.path.join(DB_DIR, fn)

        try:
            emb = np.load(path).astype("float32")
            db.append((name, emb))
        except Exception as e:
            print(f"   ⚠️ No se pudo leer {path}: {e}")

    return db

# ---------------------------------------------------------
#  Distancia y reconocimiento
# ---------------------------------------------------------


def cosine_distance(a, b):
    a = a.astype("float32")
    b = b.astype("float32")
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return 1.0 - float(np.dot(a / na, b / nb))


def recognize_face(face_bgr, db_embeddings, threshold=0.6):
    """
    Devuelve (nombre, distancia) o (None, distancia_minima)
    """
    if not db_embeddings:
        return None, None

    emb = embed_face(face_bgr)
    if emb is None:
        return None, None

    best_name = None
    best_dist = 1e9

    for name, db_emb in db_embeddings:
        d = cosine_distance(emb, db_emb)
        if d < best_dist:
            best_dist = d
            best_name = name

    if best_dist < threshold:
        return best_name, best_dist
    else:
        return None, best_dist

# ---------------------------------------------------------
#  Main con FaceMesh + DeepFace
# ---------------------------------------------------------


def main():
    print("🧠 Cargando modelo DeepFace (esto tarda solo la primera vez)...")
    GLOBAL_MODEL_NAME = "Facenet512"
    try:
        DeepFace.build_model(GLOBAL_MODEL_NAME)
    except Exception:
        GLOBAL_MODEL_NAME = "VGG-Face"
        DeepFace.build_model(GLOBAL_MODEL_NAME)
    print(f"✅ Modelo cargado: {GLOBAL_MODEL_NAME}")

    db_embeddings = load_db_embeddings()
    print(f"📁 Embeddings cargados desde disco: {len(db_embeddings)} vectores")

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # Specs para la malla
    mesh_spec_unknown = mp_drawing.DrawingSpec(
        color=(0, 0, 255), thickness=1, circle_radius=1
    )  # rojo (BGR)
    mesh_spec_known = mp_drawing.DrawingSpec(
        color=(0, 255, 0), thickness=1, circle_radius=1
    )  # verde (BGR)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

    print("\n▶ Cámara abierta. Controles:")
    print("   - R : registrar rostro que está en pantalla")
    print("   - Q : salir\n")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        last_status = "No hay personas registradas"
        last_color = (0, 0, 255)
        last_is_known = False  # para el color de la malla
        last_face_crop = None

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ No se pudo leer frame de la cámara")
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            face_crop = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # Bounding box del rostro para recortar y mandar a DeepFace
                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                min_x = max(int(min(xs) * w) - 10, 0)
                max_x = min(int(max(xs) * w) + 10, w - 1)
                min_y = max(int(min(ys) * h) - 10, 0)
                max_y = min(int(max(ys) * h) + 10, h - 1)

                if max_x > min_x and max_y > min_y:
                    face_crop = frame[min_y:max_y, min_x:max_x].copy()
                    last_face_crop = face_crop.copy()

                # Reconocimiento solo cada N frames
                if face_crop is not None and db_embeddings:
                    if frame_count % RECOGNITION_INTERVAL == 0:
                        name, dist = recognize_face(face_crop, db_embeddings)
                        if name is None:
                            last_status = "Rostro desconocido"
                            last_color = (0, 0, 255)
                            last_is_known = False
                        else:
                            last_status = f"{name} (dist {dist:.2f})"
                            last_color = (0, 255, 0)
                            last_is_known = True
                elif face_crop is not None and not db_embeddings:
                    last_status = "No hay personas registradas"
                    last_color = (0, 0, 255)
                    last_is_known = False

                # Elegir color de malla según si es conocido o no
                mesh_spec = mesh_spec_known if last_is_known else mesh_spec_unknown

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mesh_spec,
                )
            else:
                last_status = "No se detectó ningún rostro"
                last_color = (0, 0, 255)
                last_is_known = False

            # Barra de estado abajo
            cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
            cv2.putText(
                frame,
                last_status,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                last_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("DeepFace + FaceMesh (demo)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("r"), ord("R")):
                # Registrar rostro actual
                if last_face_crop is None:
                    print("⚠️ No hay rostro detectado para registrar")
                    continue

                name = input("Nombre / ID para este rostro: ").strip()
                if not name:
                    print("⚠️ Nombre vacío, se cancela el registro")
                    continue

                emb = embed_face(last_face_crop)
                if emb is None:
                    print("⚠️ No se pudo obtener embedding para este rostro")
                    continue

                db_embeddings.append((name, emb))
                save_embedding(name, emb)
                print(f"✅ Rostro registrado como '{name}'")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

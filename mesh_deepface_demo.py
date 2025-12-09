import cv2
import os
import time
import numpy as np

from deepface import DeepFace
import mediapipe as mp


# ============ CONFIGURACIÓN ============

FACES_DIR = "faces_db"   # cámbialo a "dataset" si quieres
MODEL_NAME = "Facenet512"  # también puedes probar "ArcFace"
RECOGNITION_INTERVAL = 2.0  # segundos entre reconocimientos completos
DIST_THRESHOLD = 25.0       # umbral de distancia L2 (ajustar según pruebas)


# ============ CARGA DE MODELO / EMBEDDINGS ============

def build_embeddings_database():
    """
    Recorre FACES_DIR, calcula un embedding por persona
    (usando la primera imagen encontrada) y devuelve
    una lista de diccionarios {name, embedding}.
    """
    db = []

    if not os.path.exists(FACES_DIR):
        print(f"⚠️  No existe la carpeta {FACES_DIR}. Créala y añade personas.")
        return db

    print("🔄 Construyendo base de embeddings DeepFace...")
    for folder in os.listdir(FACES_DIR):
        person_path = os.path.join(FACES_DIR, folder)
        if not os.path.isdir(person_path):
            continue

        # El nombre lo sacamos directamente de la carpeta
        person_name = folder

        # Busca una imagen cualquiera dentro
        img_files = [
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not img_files:
            continue

        img_path = os.path.join(person_path, img_files[0])
        print(f"  ➜ {person_name}: usando {img_files[0]} para embedding")

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]
            emb = np.array(rep, dtype="float32")
            db.append({"name": person_name, "embedding": emb})
        except Exception as e:
            print(f"    ⚠️  Error con {img_path}: {e}")

    print(f"✅ Embeddings cargados: {len(db)} personas\n")
    return db


def recognize_face_with_db(face_bgr, db):
    """
    Recibe la ROI de la cara en BGR y la compara con todos
    los embeddings de la base. Devuelve:
      (authorized, name, distance)
    """
    if not db:
        return False, None, None

    try:
        rep = DeepFace.represent(
            img_path=face_bgr,
            model_name=MODEL_NAME,
            enforce_detection=False
        )[0]["embedding"]
    except Exception as e:
        print(f"DeepFace error: {e}")
        return False, None, None

    query_emb = np.array(rep, dtype="float32")

    best_name = None
    best_dist = 1e9

    for entry in db:
        dist = np.linalg.norm(query_emb - entry["embedding"])
        if dist < best_dist:
            best_dist = dist
            best_name = entry["name"]

    authorized = best_dist < DIST_THRESHOLD
    return authorized, best_name if authorized else None, best_dist


# ============ MEDIAPIPE FACE MESH ============

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Para dibujar las conexiones (malla)
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION


def draw_face_mesh(frame, face_landmarks, color):
    """
    Dibuja la malla de la cara con un color dado.
    """
    drawing_spec = mp_drawing.DrawingSpec(
        color=color, thickness=1, circle_radius=1
    )
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=FACE_CONNECTIONS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_spec,
    )


def landmarks_to_bbox(face_landmarks, img_width, img_height, margin=0.1):
    """
    Convierte los landmarks (normalizados) en un bounding box [x1, y1, x2, y2]
    con un pequeño margen alrededor.
    """
    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]

    min_x = max(min(xs) - margin, 0.0)
    max_x = min(max(xs) + margin, 1.0)
    min_y = max(min(ys) - margin, 0.0)
    max_y = min(max(ys) + margin, 1.0)

    x1 = int(min_x * img_width)
    x2 = int(max_x * img_width)
    y1 = int(min_y * img_height)
    y2 = int(max_y * img_height)

    return x1, y1, x2, y2


# ============ MAIN LOOP ============

def main():
    db = build_embeddings_database()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return
    print("✅ Cámara abierta.")

    last_recognition_time = 0.0
    last_auth = False
    last_name = None
    last_dist = None

    # MediaPipe FaceMesh en modo stream
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # espejo
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                # Sólo usamos la primera cara
                face_landmarks = results.multi_face_landmarks[0]

                # Bounding box a partir de los landmarks
                x1, y1, x2, y2 = landmarks_to_bbox(face_landmarks, w, h)

                # Recortamos ROI para reconocimiento
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size != 0:
                    # Reconocemos cada cierto tiempo para no ir lentos
                    now = time.time()
                    if now - last_recognition_time > RECOGNITION_INTERVAL:
                        last_recognition_time = now
                        auth, name, dist = recognize_face_with_db(face_roi, db)
                        last_auth, last_name, last_dist = auth, name, dist

                # Color según última decisión
                if last_auth:
                    color = (0, 255, 0)   # verde
                else:
                    color = (0, 0, 255)   # rojo

                # Dibujar malla
                draw_face_mesh(frame, face_landmarks, color)

                # Dibujar texto estilo "Approved face..."
                if last_auth and last_name:
                    text = f"Approved face: {last_name} (dist={last_dist:.1f})"
                elif db:
                    text = "Unknown face"
                else:
                    text = "No hay personas registradas"

                cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
                cv2.putText(
                    frame, text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
                )
            else:
                # Sin cara detectada, mostramos aviso
                cv2.putText(
                    frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                )

            cv2.imshow("DeepFace + FaceMesh Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

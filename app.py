# app.py
import os
import threading
import time
from datetime import date

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, Response
)

from db import (
    get_connection,
    crear_socio,
    obtener_socio_por_cedula,
    obtener_socio_por_id,
    obtener_membresia_activa,
    registrar_acceso,
    listar_socios,
    buscar_socios,
    obtener_socio_completo,
    obtener_membresias_socio,
    obtener_membresia_activa_detalle,
    crear_membresia,
)

app = Flask(__name__)
app.secret_key = "cambia_esta_clave_super_secreta"

# ==========================
#  CÁMARA GLOBAL
# ==========================

cap_lock = threading.Lock()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara en el índice 0")
else:
    print("✅ Cámara abierta en el índice 0")

last_frame = None      # solo para mostrar en /video_feed
last_face_roi = None   # bounding box del rostro para la vista


def get_frame():
    """
    Lee un frame de la cámara de forma segura.
    Si la cámara está cerrada intenta reabrirla.
    """
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("⚠️ No se pudo reabrir la cámara")
                return None

        ret, frame = cap.read()

    if not ret:
        print("⚠️ cap.read() devolvió False (no frame)")
        return None

    return frame


# ==========================
#  MEDIAPIPE
# ==========================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ==========================
#  DEEPFACE / EMBEDDINGS
#   (MISMO ESTILO QUE TU SCRIPT DEMO)
# ==========================

DB_EMB_DIR = "deepface_db"
os.makedirs(DB_EMB_DIR, exist_ok=True)

RECOGNITION_THRESHOLD = 0.40  # igual que el script que funcionaba

print("🧠 Cargando modelo DeepFace (esto tarda solo la primera vez)...")
GLOBAL_MODEL_NAME = "Facenet512"
try:
    DeepFace.build_model(GLOBAL_MODEL_NAME)
except Exception:
    GLOBAL_MODEL_NAME = "VGG-Face"
    DeepFace.build_model(GLOBAL_MODEL_NAME)
print(f"✅ Modelo DeepFace cargado: {GLOBAL_MODEL_NAME}")


def embed_face(face_bgr):
    """
    Devuelve el embedding de un rostro como np.array(float32).
    Es el MISMO código que usabas en tu script que funcionaba.
    """
    try:
        try:
            reps = DeepFace.represent(
                face_bgr,
                model_name=GLOBAL_MODEL_NAME,
                enforce_detection=False,
                detector_backend="opencv",
            )
        except TypeError:
            reps = DeepFace.represent(
                face_bgr,
                enforce_detection=False,
                detector_backend="opencv",
            )

        if not reps:
            return None

        rep = reps[0]

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


def save_embedding(cedula, emb):
    """
    Guarda embedding en disco con prefijo la cédula: cedula__timestamp.npy
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = f"{cedula}__{ts}.npy"
    path = os.path.join(DB_EMB_DIR, fn)
    np.save(path, emb)


def load_db_embeddings():
    """
    Lee todos los .npy en deepface_db y construye [(cedula, emb), ...].
    """
    db = []
    if not os.path.isdir(DB_EMB_DIR):
        return db

    for fn in os.listdir(DB_EMB_DIR):
        if not fn.lower().endswith(".npy"):
            continue

        cedula = fn.split("__", 1)[0]
        path = os.path.join(DB_EMB_DIR, fn)

        try:
            emb = np.load(path).astype("float32")
            db.append((cedula, emb))
        except Exception as e:
            print(f"   ⚠️ No se pudo leer {path}: {e}")

    return db


def cosine_distance(a, b):
    a = a.astype("float32")
    b = b.astype("float32")
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return 1.0 - float(np.dot(a / na, b / nb))


def recognize_face_by_embeddings(face_bgr, db_embeddings, threshold=RECOGNITION_THRESHOLD):
    """
    Igual que recognize_face de tu script:
    devuelve (cedula, distancia) o (None, distancia_minima)
    """
    if not db_embeddings:
        return None, None

    emb = embed_face(face_bgr)
    if emb is None:
        return None, None

    best_cedula = None
    best_dist = 1e9

    for cedula, db_emb in db_embeddings:
        d = cosine_distance(emb, db_emb)
        if d < best_dist:
            best_dist = d
            best_cedula = cedula

    if best_dist < threshold:
        return best_cedula, best_dist
    else:
        return None, best_dist


# cargamos embeddings al arrancar
db_embeddings = load_db_embeddings()
print(f"📁 Embeddings cargados desde disco: {len(db_embeddings)} vectores")

# ==========================
#  ESTADO GLOBAL DEL KIOSKO
# ==========================

kiosk_status_lock = threading.Lock()
kiosk_status = {
    "permitido": None,  # True / False / None
    "mensaje": "No se detectó ningún rostro",
    "nombre": None,
    "membresia": None,
    "motivo": None,
    "distancia": None,
}

# ==========================
#  VIDEO STREAMING (para /camara)
# ==========================


def gen_frames():
    """
    Generador para /video_feed.
    Dibuja FaceMesh y colorea la malla según kiosk_status.
    """
    global last_frame, last_face_roi

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            frame = get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                min_x = max(int(min(xs) * w) - 10, 0)
                max_x = min(int(max(xs) * w) + 10, w - 1)
                min_y = max(int(min(ys) * h) - 10, 0)
                max_y = min(int(max(ys) * h) + 10, h - 1)

                if max_x > min_x and max_y > min_y:
                    last_face_roi = (min_x, min_y, max_x, max_y)
                else:
                    last_face_roi = None

                with kiosk_status_lock:
                    permitido = kiosk_status.get("permitido")

                if permitido is True:
                    color = (0, 255, 0)      # verde
                elif permitido is False:
                    color = (0, 0, 255)      # rojo
                else:
                    color = (255, 255, 255)  # blanco

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=color, thickness=1, circle_radius=1
                    ),
                )
            else:
                last_face_roi = None

            last_frame = frame.copy()

            ret2, buffer = cv2.imencode(".jpg", frame)
            if not ret2:
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            time.sleep(0.03)


def get_face_from_last_frame(loose=False):
    """
    Versión que imita al script que sí funciona:
    - Toma un frame nuevo de la cámara
    - Usa FaceMesh para encontrar el bounding box
    - Devuelve el recorte de la cara (BGR)
    El parámetro 'loose' se ignora, se deja solo para compatibilidad.
    """
    frame = get_frame()
    if frame is None:
        return None

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]
    min_x = max(int(min(xs) * w) - 10, 0)
    max_x = min(int(max(xs) * w) + 10, w - 1)
    min_y = max(int(min(ys) * h) - 10, 0)
    max_y = min(int(max(ys) * h) + 10, h - 1)

    if max_x <= min_x or max_y <= min_y:
        return None

    face_crop = frame[min_y:max_y, min_x:max_x].copy()
    if face_crop.size == 0:
        return None

    return face_crop

# ==========================
#  LOGIN / SESIÓN
# ==========================


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == "Jhoan" and password == "123456789$":
            session["admin_username"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Usuario o contraseña incorrectos", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def requiere_login(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "admin_username" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper


# ==========================
#  DASHBOARD
# ==========================

@app.route("/")
@requiere_login
def dashboard():
    conn = get_connection()
    conn.row_factory = None
    cur = conn.cursor()

    # --- obtener la fecha "de hoy" según SQLite (localtime) ---
    cur.execute("SELECT DATE('now','localtime')")
    hoy = cur.fetchone()[0]

    # --- contar SOLO socios permitidos, ÚNICOS por día ---
    cur.execute("""
        SELECT COUNT(DISTINCT socio_id)
        FROM accesos
        WHERE DATE(momento) = DATE('now','localtime')
          AND permitido = 1
          AND socio_id IS NOT NULL
    """)
    accesos_hoy = cur.fetchone()[0]

    # --- listar SOLO accesos permitidos (últimos 20), sin desconocidos ---
    cur.execute("""
        SELECT a.momento, a.permitido, a.motivo,
               s.nombre, s.apellido, s.cedula,
               tm.codigo AS tipo_codigo, tm.descripcion AS tipo_descripcion
        FROM accesos a
        LEFT JOIN socios s ON s.id = a.socio_id
        LEFT JOIN tipo_membresia tm ON tm.codigo = a.tipo_membresia_codigo
        WHERE a.permitido = 1
          AND a.socio_id IS NOT NULL
        ORDER BY a.momento DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    conn.close()

    accesos = []
    for r in rows:
        accesos.append({
            "momento": r[0],
            "permitido": r[1],
            "motivo": r[2],
            "nombre": r[3],
            "apellido": r[4],
            "cedula": r[5],
            "tipo_codigo": r[6],
            "tipo_descripcion": r[7],
        })

    return render_template(
        "dashboard.html",
        hoy=hoy,
        accesos_hoy=accesos_hoy,
        accesos=accesos,
    )

# ==========================
#  USUARIOS (socios)
# ==========================


@app.route("/socios")
@requiere_login
def socios_listado():
    q = request.args.get("q", "").strip()
    if q:
        socios = buscar_socios(q)
    else:
        socios = listar_socios()
    return render_template("socios_listado.html", socios=socios, q=q)


@app.route("/socios/nuevo", methods=["GET", "POST"])
@requiere_login
def socios_nuevo():
    # tipos de membresía para el combo (MENSUAL, QUINCENA, TIQUETERA, DIARIO, etc)
    conn = get_connection()
    conn.row_factory = None
    cur = conn.cursor()
    cur.execute("SELECT codigo, descripcion FROM tipo_membresia")
    tipos = cur.fetchall()
    conn.close()

    if request.method == "POST":
        nombre = request.form.get("nombre", "").strip()
        apellido = request.form.get("apellido", "").strip()
        cedula = request.form.get("cedula", "").strip()
        edad = request.form.get("edad") or None
        tipo_codigo = request.form.get("tipo_codigo")
        valor_pagado = request.form.get("valor_pagado") or 0

        if not nombre or not apellido or not cedula:
            flash("Nombre, apellido y cédula son obligatorios", "danger")
            return render_template("socios_nuevo.html", tipos=tipos)

        if not tipo_codigo:
            flash("Debes seleccionar un tipo de membresía inicial", "danger")
            return render_template("socios_nuevo.html", tipos=tipos)

        socio_existente = obtener_socio_por_cedula(cedula)
        if socio_existente:
            flash("Ya existe un usuario con esa cédula", "danger")
            return render_template("socios_nuevo.html", tipos=tipos)

        socio_id = crear_socio(nombre, apellido, cedula, edad)
        crear_membresia(socio_id, tipo_codigo, float(valor_pagado))

        flash("Usuario creado correctamente. Ahora captura su rostro.", "success")
        return redirect(url_for("socio_rostro", socio_id=socio_id))

    return render_template("socios_nuevo.html", tipos=tipos)


@app.route("/socios/<int:socio_id>")
@requiere_login
def socio_detalle(socio_id):
    socio = obtener_socio_completo(socio_id)
    if not socio:
        flash("Usuario no encontrado", "danger")
        return redirect(url_for("socios_listado"))

    membresia_activa = obtener_membresia_activa_detalle(socio_id)

    return render_template(
        "socio_detalle.html",
        socio=socio,
        membresia_activa=membresia_activa,
    )


@app.route("/socios/<int:socio_id>/rostro")
@requiere_login
def socio_rostro(socio_id):
    socio = obtener_socio_por_id(socio_id)
    if not socio:
        flash("Usuario no encontrado", "danger")
        return redirect(url_for("socios_listado"))

    return render_template("socio_rostro.html", socio=socio)


@app.route("/api/socios/<int:socio_id>/capturar-rostro", methods=["POST"])
@requiere_login
def api_capturar_rostro(socio_id):
    """
    Captura / actualiza el rostro de un usuario usando la cámara.
    """
    socio = obtener_socio_por_id(socio_id)
    if not socio:
        return jsonify(ok=False, mensaje="Usuario no encontrado")

    global db_embeddings
    face = get_face_from_last_frame(loose=True)
    if face is None or face.size == 0:
        return jsonify(
            ok=False,
            mensaje="No se detectó rostro. Acércate y centra la cara."
        )

    emb = embed_face(face)
    if emb is None:
        return jsonify(ok=False, mensaje="No se pudo obtener embedding del rostro")

    cedula = socio["cedula"]

    # Borrar embeddings viejos de esta cédula
    try:
        for fn in os.listdir(DB_EMB_DIR):
            if fn.startswith(f"{cedula}__") and fn.endswith(".npy"):
                try:
                    os.remove(os.path.join(DB_EMB_DIR, fn))
                except OSError:
                    pass
    except FileNotFoundError:
        pass

    save_embedding(cedula, emb)
    db_embeddings = load_db_embeddings()

    return jsonify(ok=True, mensaje="Rostro capturado / actualizado correctamente.")


@app.route("/socios/<int:socio_id>/editar", methods=["GET", "POST"])
@requiere_login
def socio_editar(socio_id):
    socio = obtener_socio_por_id(socio_id)
    if not socio:
        flash("Usuario no encontrado", "danger")
        return redirect(url_for("socios_listado"))

    if request.method == "POST":
        nombre = request.form.get("nombre", "").strip()
        apellido = request.form.get("apellido", "").strip()
        cedula = request.form.get("cedula", "").strip()
        edad = request.form.get("edad") or None
        activo = int(request.form.get("activo", "1"))

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE socios SET nombre=?, apellido=?, cedula=?, edad=?, activo=?
            WHERE id=?
        """, (nombre, apellido, cedula, edad, activo, socio_id))
        conn.commit()
        conn.close()

        flash("Usuario actualizado", "success")
        return redirect(url_for("socio_detalle", socio_id=socio_id))

    return render_template("socio_editar.html", socio=socio)


@app.route("/socios/<int:socio_id>/eliminar", methods=["POST"])
@requiere_login
def socio_eliminar(socio_id):
    socio = obtener_socio_por_id(socio_id)
    if not socio:
        flash("Usuario no encontrado", "danger")
        return redirect(url_for("socios_listado"))

    cedula = socio["cedula"]

    # 1) borrar embeddings de esa cédula
    try:
        for fn in os.listdir(DB_EMB_DIR):
            if fn.startswith(f"{cedula}__") and fn.endswith(".npy"):
                try:
                    os.remove(os.path.join(DB_EMB_DIR, fn))
                except OSError:
                    pass
    except FileNotFoundError:
        pass

    # 2) borrar registros dependientes en BD (accesos, membresías, socio)
    conn = get_connection()
    cur = conn.cursor()

    # si tienes claves foráneas sin ON DELETE CASCADE,
    # hay que borrar primero hijos:
    cur.execute("DELETE FROM accesos WHERE socio_id = ?", (socio_id,))
    cur.execute("DELETE FROM membresias WHERE socio_id = ?", (socio_id,))
    cur.execute("DELETE FROM socios WHERE id = ?", (socio_id,))

    conn.commit()
    conn.close()

    # 3) recargar embeddings en memoria
    global db_embeddings
    db_embeddings = load_db_embeddings()

    flash("Usuario eliminado correctamente", "success")
    return redirect(url_for("socios_listado"))


# ==========================
#  MEMBRESÍA (renovar)
# ==========================

@app.route("/socios/<int:socio_id>/membresia/nueva", methods=["GET", "POST"])
@requiere_login
def membresia_nueva(socio_id):
    socio = obtener_socio_por_id(socio_id)
    if not socio:
        flash("Usuario no encontrado", "danger")
        return redirect(url_for("socios_listado"))

    conn = get_connection()
    conn.row_factory = None
    cur = conn.cursor()
    cur.execute("SELECT codigo, descripcion FROM tipo_membresia")
    tipos = cur.fetchall()
    conn.close()

    if request.method == "POST":
        tipo_codigo = request.form.get("tipo_codigo")
        valor_pagado = request.form.get("valor_pagado") or 0

        if not tipo_codigo:
            flash("Debes seleccionar tipo de membresía", "danger")
            return render_template("membresia_nueva.html", socio=socio, tipos=tipos)

        crear_membresia(socio_id, tipo_codigo, float(valor_pagado))
        flash("Membresía renovada", "success")
        return redirect(url_for("socio_detalle", socio_id=socio_id))

    return render_template("membresia_nueva.html", socio=socio, tipos=tipos)


# ==========================
#  KIOSKO CÁMARA + DEEPFACE
# ==========================

@app.route("/camara")
@requiere_login
def camara():
    return render_template("camara.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/registrar-con-rostro", methods=["POST"])
@requiere_login
def api_registrar_con_rostro():
    """
    Crea usuario + guarda embedding con la cámara (modo registro rápido).
    """
    data = request.get_json(silent=True) or {}
    nombre = data.get("nombre", "").strip()
    apellido = data.get("apellido", "").strip()
    cedula = data.get("cedula", "").strip()
    edad = data.get("edad") or None

    if not nombre or not apellido or not cedula:
        return jsonify(ok=False, mensaje="Nombre, apellido y cédula son obligatorios")

    if obtener_socio_por_cedula(cedula):
        return jsonify(ok=False, mensaje="Ya existe un usuario con esa cédula")

    global db_embeddings
    face = get_face_from_last_frame(loose=True)
    if face is None or face.size == 0:
        return jsonify(ok=False, mensaje="No se detectó rostro. Acércate y centra la cara.")

    emb = embed_face(face)
    if emb is None:
        return jsonify(ok=False, mensaje="No se pudo obtener embedding del rostro")

    socio_id = crear_socio(nombre, apellido, cedula, edad)
    save_embedding(cedula, emb)
    db_embeddings = load_db_embeddings()

    return jsonify(ok=True, mensaje=f"Usuario '{nombre} {apellido}' registrado con éxito", socio_id=socio_id)


@app.route("/api/reconocer-rostro", methods=["POST"])
def api_reconocer_rostro():
    """
    Usa la cámara → rostro → embeddings → verifica membresía → registra acceso.
    Además actualiza kiosk_status para colorear la malla y mostrar info en /camara.
    """
    global db_embeddings, kiosk_status

    print("➡️ /api/reconocer-rostro llamado")

    face = get_face_from_last_frame(loose=False)
    if face is None or face.size == 0:
        with kiosk_status_lock:
            kiosk_status.update({
                "permitido": None,
                "mensaje": "No se detectó ningún rostro",
                "nombre": None,
                "membresia": None,
                "motivo": "Sin rostro en la imagen",
                "distancia": None,
            })
        resp = dict(ok=False, mensaje="No se detectó ningún rostro")
        print("   Sin rostro para reconocimiento")
        return jsonify(resp)

    if not db_embeddings:
        with kiosk_status_lock:
            kiosk_status.update({
                "permitido": None,
                "mensaje": "No hay usuarios registrados con rostro",
                "nombre": None,
                "membresia": None,
                "motivo": "Base de rostros vacía",
                "distancia": None,
            })
        return jsonify(ok=False, mensaje="No hay usuarios registrados con rostro. Registra primero.")

    cedula, dist = recognize_face_by_embeddings(face, db_embeddings)
    print(f"   Resultado DeepFace: cedula={cedula}, dist={dist}")

    if cedula is None:
        registrar_acceso(None, False, "Rostro desconocido", None)
        resp = dict(
            ok=True,
            permitido=False,
            mensaje="Rostro desconocido",
            nombre=None,
            membresia=None,
            motivo="Rostro no registrado en el sistema.",
            distancia=dist,
        )
        with kiosk_status_lock:
            kiosk_status.update(resp)
        print(f"   Estado kiosko actualizado: {resp}")
        return jsonify(resp)

    socio = obtener_socio_por_cedula(cedula)
    if not socio:
        registrar_acceso(None, False,
                         "Cédula reconocida pero usuario no existe", None)
        resp = dict(
            ok=True,
            permitido=False,
            mensaje="Usuario no encontrado para esta cédula",
            nombre=None,
            membresia=None,
            motivo=f"Cédula {cedula} no está en la tabla de usuarios.",
            distancia=dist,
        )
        with kiosk_status_lock:
            kiosk_status.update(resp)
        print(f"   Estado kiosko actualizado: {resp}")
        return jsonify(resp)

    socio_id = socio["id"]
    nombre_completo = f"{socio['nombre']} {socio['apellido']}"

    mem = obtener_membresia_activa_detalle(socio_id)
    if not mem:
        registrar_acceso(socio_id, False, "Sin membresía activa", None)
        resp = dict(
            ok=True,
            permitido=False,
            mensaje=f"Membresía vencida o inexistente para {nombre_completo}",
            nombre=nombre_completo,
            membresia="Sin membresía activa",
            motivo="Debe renovar su plan.",
            distancia=dist,
        )
        with kiosk_status_lock:
            kiosk_status.update(resp)
        print(f"   Estado kiosko actualizado: {resp}")
        return jsonify(resp)

    cod = mem["tipo_codigo"]
    desc = mem["tipo_descripcion"]
    fin = mem["fecha_fin"]
    texto_m = f"{cod} - {desc} (hasta {fin})"

    registrar_acceso(socio_id, True, "Acceso permitido", cod)

    resp = dict(
        ok=True,
        permitido=True,
        mensaje=f"Acceso autorizado: {nombre_completo}",
        nombre=nombre_completo,
        membresia=texto_m,
        motivo="Membresía activa",
        distancia=dist,
    )
    with kiosk_status_lock:
        kiosk_status.update(resp)

    print(f"   Estado kiosko actualizado: {resp}")
    return jsonify(resp)


# ==========================
#  MAIN
# ==========================

if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        with cap_lock:
            if cap is not None and cap.isOpened():
                cap.release()

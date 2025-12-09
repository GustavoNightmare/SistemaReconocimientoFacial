# db.py
import sqlite3
from datetime import date, timedelta

DB_PATH = "gimnasio.db"


def get_connection():
    """
    Devuelve una conexión SQLite lista para usar.
    - timeout alto para evitar 'database is locked'
    - check_same_thread=False porque Flask usa hilos
    - row_factory=Row para acceder por nombre de columna
    """
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ==========================
#  INICIALIZAR BASE DE DATOS
# ==========================

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("PRAGMA foreign_keys = ON;")

    # --- tabla socios ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS socios (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre          TEXT NOT NULL,
            apellido        TEXT NOT NULL,
            cedula          TEXT NOT NULL UNIQUE,
            edad            INTEGER,
            activo          INTEGER NOT NULL DEFAULT 1,
            fecha_creacion  TEXT NOT NULL DEFAULT (date('now'))
        );
    """)

    # --- tipos de membresía ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tipo_membresia (
            codigo                  TEXT PRIMARY KEY,
            descripcion             TEXT NOT NULL,
            dias_duracion           INTEGER NOT NULL,
            tickets_totales_default INTEGER,
            permite_ilimitado       INTEGER NOT NULL DEFAULT 0
        );
    """)

    # --- membresías ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS membresias (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            socio_id             INTEGER NOT NULL,
            tipo_membresia_codigo TEXT NOT NULL,
            fecha_inicio         TEXT NOT NULL,
            fecha_fin            TEXT NOT NULL,
            valor_pagado         REAL NOT NULL DEFAULT 0,
            tickets_totales      INTEGER NOT NULL DEFAULT 0,
            tickets_restantes    INTEGER NOT NULL DEFAULT 0,
            activo               INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(socio_id) REFERENCES socios(id),
            FOREIGN KEY(tipo_membresia_codigo) REFERENCES tipo_membresia(codigo)
        );
    """)

    # --- registros de accesos ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS accesos (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            socio_id              INTEGER,
            permitido             INTEGER NOT NULL,
            motivo                TEXT,
            tipo_membresia_codigo TEXT,
            momento               TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(socio_id) REFERENCES socios(id),
            FOREIGN KEY(tipo_membresia_codigo) REFERENCES tipo_membresia(codigo)
        );
    """)

    # Insertar tipos de membresía base si no hay ninguno
    # Insertar tipos de membresía base si no hay ninguno
    cur.execute("SELECT COUNT(*) AS c FROM tipo_membresia;")
    if cur.fetchone()["c"] == 0:
        cur.executemany("""
            INSERT INTO tipo_membresia
            (codigo, descripcion, dias_duracion, tickets_totales_default, permite_ilimitado)
            VALUES (?, ?, ?, ?, ?);
        """, [
            # Mensualidad ilimitada (como la tenías)
            ("MENSUAL", "Mensual ilimitada", 30, None, 1),

            # Quincena ilimitada
            ("QUINCENA", "Quincena ilimitada", 15, None, 1),

            # Tiquetera: X entradas, vence en algunos días
            ("TIQUETERA", "Plan por tickets (10 entradas)", 90, 10, 0),
        ])

    conn.commit()
    conn.close()


init_db()  # se ejecuta al importar el módulo


# ==========================
#  SOCIOS
# ==========================

def crear_socio(nombre, apellido, cedula, edad):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO socios (nombre, apellido, cedula, edad)
        VALUES (?, ?, ?, ?);
    """, (nombre, apellido, cedula, edad))
    socio_id = cur.lastrowid
    conn.commit()
    conn.close()
    return socio_id


def obtener_socio_por_cedula(cedula):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM socios WHERE cedula = ?;", (cedula,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def obtener_socio_por_id(socio_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM socios WHERE id = ?;", (socio_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def listar_socios():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, 
               COALESCE(tm.descripcion, 'Sin membresía') AS tipo_membresia
        FROM socios s
        LEFT JOIN membresias m 
            ON m.socio_id = s.id AND m.activo = 1
        LEFT JOIN tipo_membresia tm 
            ON tm.codigo = m.tipo_membresia_codigo
        ORDER BY s.apellido, s.nombre;
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def buscar_socios(q):
    like = f"%{q}%"
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*,
               COALESCE(tm.descripcion, 'Sin membresía') AS tipo_membresia
        FROM socios s
        LEFT JOIN membresias m 
            ON m.socio_id = s.id AND m.activo = 1
        LEFT JOIN tipo_membresia tm 
            ON tm.codigo = m.tipo_membresia_codigo
        WHERE s.nombre LIKE ? OR s.apellido LIKE ? OR s.cedula LIKE ?
        ORDER BY s.apellido, s.nombre;
    """, (like, like, like))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def obtener_socio_completo(socio_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM socios WHERE id = ?;", (socio_id,))
    socio = cur.fetchone()
    conn.close()
    return dict(socio) if socio else None


# ==========================
#  MEMBRESÍAS
# ==========================

def crear_membresia(socio_id, tipo_codigo, valor_pagado):
    """
    Crea o renueva la membresía de un socio.

    - Calcula fecha_fin a partir de dias_duracion.
    - Si la membresía es ilimitada => tickets_totales = tickets_restantes = 0.
    - Si tiene tickets_por_defecto => se usan esos para ambos campos.
    - Desactiva cualquier membresía previa del socio.
    """
    conn = get_connection()
    cur = conn.cursor()

    # info del tipo de membresía
    cur.execute("""
        SELECT dias_duracion, tickets_totales_default, permite_ilimitado
        FROM tipo_membresia
        WHERE codigo = ?;
    """, (tipo_codigo,))
    tipo = cur.fetchone()
    if not tipo:
        conn.close()
        raise ValueError(f"Tipo de membresía '{tipo_codigo}' no existe")

    dias = tipo["dias_duracion"]
    tickets_default = tipo["tickets_totales_default"]
    ilimitado = bool(tipo["permite_ilimitado"])

    hoy = date.today()
    fecha_inicio = hoy.isoformat()
    fecha_fin = (hoy + timedelta(days=dias)).isoformat()

    if ilimitado:
        tickets_totales = 0
        tickets_restantes = 0
    else:
        tickets_totales = tickets_default if tickets_default is not None else 0
        tickets_restantes = tickets_totales

    # desactivar membresías anteriores
    cur.execute("""
        UPDATE membresias
        SET activo = 0
        WHERE socio_id = ?;
    """, (socio_id,))

    # insertar nueva membresía
    cur.execute("""
        INSERT INTO membresias
        (socio_id, tipo_membresia_codigo, fecha_inicio, fecha_fin,
         valor_pagado, tickets_totales, tickets_restantes, activo)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1);
    """, (
        socio_id,
        tipo_codigo,
        fecha_inicio,
        fecha_fin,
        valor_pagado,
        tickets_totales,
        tickets_restantes,
    ))

    conn.commit()
    conn.close()


def obtener_membresia_activa_detalle(socio_id):
    """
    Devuelve dict con la membresía activa y no vencida del socio,
    incluyendo datos del tipo de membresía.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT m.*,
               tm.codigo       AS tipo_codigo,
               tm.descripcion  AS tipo_descripcion
        FROM membresias m
        JOIN tipo_membresia tm
          ON tm.codigo = m.tipo_membresia_codigo
        WHERE m.socio_id = ?
          AND m.activo = 1
          AND date(m.fecha_fin) >= date('now')
        ORDER BY m.fecha_fin DESC
        LIMIT 1;
    """, (socio_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def obtener_membresia_activa(socio_id):
    """
    Versión simple que solo devuelve la fila de membresías.
    (La usa el dashboard antiguo; la dejamos por compatibilidad.)
    """
    mem = obtener_membresia_activa_detalle(socio_id)
    return mem


def obtener_membresias_socio(socio_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT m.*, tm.descripcion AS tipo_descripcion
        FROM membresias m
        JOIN tipo_membresia tm ON tm.codigo = m.tipo_membresia_codigo
        WHERE m.socio_id = ?
        ORDER BY m.fecha_inicio DESC;
    """, (socio_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ==========================
#  ACCESOS (LOG KIOSKO)
# ==========================

def registrar_acceso(socio_id, permitido, motivo, tipo_membresia_codigo):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO accesos (socio_id, permitido, motivo, tipo_membresia_codigo)
        VALUES (?, ?, ?, ?);
    """, (socio_id, int(bool(permitido)), motivo, tipo_membresia_codigo))
    conn.commit()
    conn.close()

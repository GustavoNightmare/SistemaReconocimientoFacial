import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara en el índice 0")
    exit()

print("✅ Cámara abierta, presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer frame")
        break

    cv2.imshow("Test Cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import os

# === 1. Carica il modello CNN allenato su SignMNIST ===

MODEL_PATH = "asl_signmnist_final.keras"  # modifica se è in un'altra cartella

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modello non trovato: {MODEL_PATH}\n"
        f"Esegui prima train_model.py per addestrare e salvare il modello."
    )

print(f"[INFO] Carico modello da: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Modello caricato.")

# === 2. Mapping classi → lettere ===
index_to_letter = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y"
}

# === 3. Preprocessing: ROI a sinistra + blur + Otsu + maschera su gray ===

def preprocess_frame(frame_bgr):
    """
    Estrae una ROI a sinistra, segmenta la mano dallo sfondo con Otsu,
    e applica la maschera all'immagine in scala di grigi, mantenendo
    i dettagli interni come in SignMNIST.
    """
    h, w, _ = frame_bgr.shape

    roi_size = min(h, w) // 3

    # ROI posizionata sulla sinistra
    cx = int(w * 0.25)  # 25% larghezza → sinistra
    cy = h // 2         # centro verticale

    x1 = cx - roi_size // 2
    y1 = cy - roi_size // 2
    x2 = cx + roi_size // 2
    y2 = cy + roi_size // 2

    # Clipping ai bordi
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    roi = frame_bgr[y1:y2, x1:x2]

    # 1) Gray originale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2) Blur leggero per togliere rumore
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Otsu per separare mano / sfondo
    _, mask = cv2.threshold(
        gray_blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4) Assicuriamoci che la mano sia bianca nella maschera
    h_m, w_m = mask.shape
    if mask[h_m // 2, w_m // 2] == 0:
        mask = 255 - mask

    # 5) Pulizia maschera (rimuove puntini e buchi)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Applica la maschera al gray originale (mano grigia, sfondo nero)
    hand = cv2.bitwise_and(gray, gray, mask=mask)

    # 7) Equalizzazione per stabilizzare la luminosità (come prima)
    hand = cv2.equalizeHist(hand)

    # 8) Resize a 28×28
    hand = cv2.resize(hand, (28, 28), interpolation=cv2.INTER_AREA)

    # 9) Normalizza
    hand = hand.astype("float32") / 255.0
    hand = np.expand_dims(hand, axis=-1)   # (28, 28, 1)
    hand = np.expand_dims(hand, axis=0)    # (1, 28, 28, 1)

    return hand, (x1, y1, x2, y2)


# === 4. TTS ===

engine = pyttsx3.init()

def speak(text: str):
    print("[TTS]", text)
    engine.say(text)
    engine.runAndWait()


# === 5. Loop Webcam ===

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Impossibile aprire la webcam.")

print("[INFO] Webcam aperta.")
print("COMANDI:")
print("  SPAZIO = cattura lettera")
print("  BACKSPACE = cancella ultima")
print("  ENTER = leggi il testo")
print("  ESC = esci")

current_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERRORE] Webcam non risponde.")
        break

    h, w, _ = frame.shape
    roi_size = min(h, w) // 3

    # Rettangolo ROI a sinistra (stessa logica del preprocess)
    cx = int(w * 0.25)
    cy = h // 2

    x1 = cx - roi_size // 2
    y1 = cy - roi_size // 2
    x2 = cx + roi_size // 2
    y2 = cy + roi_size // 2

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.putText(display_frame, f"Testo: {current_text}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("ASL Webcam Demo", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == 32:  # SPAZIO
        inp, _ = preprocess_frame(frame)

        # Debug: mostra l'immagine 28×28 che va al modello
        debug_img = inp[0, :, :, 0]
        debug_vis = cv2.resize(debug_img, (280, 280),
                               interpolation=cv2.INTER_NEAREST)
        cv2.imshow("ROI pre-processata", debug_vis)

        # Predizione
        probs = model.predict(inp, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        if pred_idx in index_to_letter:
            letter = index_to_letter[pred_idx]
            current_text += letter
            print(f"[PRED] → {letter}  (classe {pred_idx})")
        else:
            print(f"[PRED] Classe sconosciuta: {pred_idx}")

    elif key == 8:  # BACKSPACE
        if current_text:
            current_text = current_text[:-1]

    elif key == 13:  # ENTER
        if current_text.strip():
            speak(current_text)
            current_text = ""

cap.release()
cv2.destroyAllWindows()
print("[INFO] Demo terminata.")

import os
import cv2
import torch
import numpy as np
import onnxruntime as ort
from lib.core.general import non_max_suppression

def resize_unscale(img, new_shape=(640, 640), color=114):
    """
    Redimensionne une image en gardant le ratio d'aspect et ajoute un padding
    pour qu'elle corresponde à la nouvelle forme tout en restant centrée.

    Args:
        img (numpy.ndarray): Image à redimensionner.
        new_shape (tuple): Taille cible (hauteur, largeur).
        color (int): Valeur des pixels pour le remplissage (par défaut 114, gris).

    Returns:
        tuple: 
            - canvas (numpy.ndarray): Image redimensionnée avec padding.
            - r (float): Ratio de redimensionnement.
            - dw (int): Décalage horizontal (padding à gauche).
            - dh (int): Décalage vertical (padding en haut).
            - new_unpad_w (int): Largeur de l'image sans padding.
            - new_unpad_h (int): Hauteur de l'image sans padding.
    """
    shape = img.shape[:2]  # Taille actuelle de l'image (hauteur, largeur)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Créer une image vide (canvas) de la nouvelle taille
    canvas = np.zeros((new_shape[0], new_shape[1], 3), dtype=np.uint8)
    canvas[:] = color

    # Calculer le ratio de redimensionnement
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Dimensions sans padding
    new_unpad_w = int(round(shape[1] * r))
    new_unpad_h = int(round(shape[0] * r))

    # Calculer le padding (horizontal et vertical)
    pad_w = new_shape[1] - new_unpad_w
    pad_h = new_shape[0] - new_unpad_h

    dw = pad_w // 2
    dh = pad_h // 2

    # Redimensionner l'image à la nouvelle taille sans padding
    resized_img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_AREA)

    # Ajouter l'image redimensionnée au centre du canvas
    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = resized_img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h


def process_video(input_video, output_video_dir, weight="yolop-640-640.onnx"):
    # Créer une session ONNX
    ort.set_default_logger_severity(4)
    onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(onnx_path)
    print(f"Loaded ONNX model: {onnx_path}")

    # Charger la vidéo d'entrée
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Définir les encodeurs pour les vidéos de sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_detect = cv2.VideoWriter(os.path.join(output_video_dir, 'detect_output.mp4'), fourcc, fps, (width, height))
    output_da = cv2.VideoWriter(os.path.join(output_video_dir, 'da_output.mp4'), fourcc, fps, (width, height))
    output_ll = cv2.VideoWriter(os.path.join(output_video_dir, 'll_output.mp4'), fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo

        # Prétraitement de la frame
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(frame, (640, 640))
        img = canvas.astype(np.float32) / 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        # Inférence
        det_out, da_seg_out, ll_seg_out = ort_session.run(
            ['det_out', 'drive_area_seg', 'lane_line_seg'],
            input_feed={"images": img}
        )

        # Détection d'objets
        det_out = torch.from_numpy(det_out).float()
        boxes = non_max_suppression(det_out)[0]
        if boxes is not None:
            boxes = boxes.cpu().numpy().astype(np.float32)
            # Mise à l'échelle des coordonnées vers la taille originale
            boxes[:, 0] -= dw
            boxes[:, 1] -= dh
            boxes[:, 2] -= dw
            boxes[:, 3] -= dh
            boxes[:, :4] /= r

        # Générer les frames pour chaque tâche
        frame_detect = frame.copy()
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2, conf, label = map(int, box)
                cv2.rectangle(frame_detect, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Segmentation
        da_seg_mask = np.argmax(da_seg_out, axis=1)[0][dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0][dh:dh + new_unpad_h, dw:dw + new_unpad_w]

        da_seg_mask = cv2.resize((da_seg_mask * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
        ll_seg_mask = cv2.resize((ll_seg_mask * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)

        frame_da = cv2.addWeighted(frame, 0.7, cv2.cvtColor(da_seg_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        frame_ll = cv2.addWeighted(frame, 0.7, cv2.cvtColor(ll_seg_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

        # Écrire les frames dans les fichiers de sortie
        output_detect.write(frame_detect)
        output_da.write(frame_da)
        output_ll.write(frame_ll)

    # Libérer les ressources
    cap.release()
    output_detect.release()
    output_da.release()
    output_ll.release()
    print(f"Processing complete. Videos saved in {output_video_dir}")

if __name__ == "__main__":
    process_video(
        input_video="./inference/videos/1.mp4",
        output_video_dir="./output",
        weight="yolop-640-640.onnx"
    )

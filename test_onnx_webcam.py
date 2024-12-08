import cv2
import torch
import numpy as np

# Assurez-vous que votre modèle est bien chargé sur le bon device (CPU ou CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MCnet(YOLOP)  # Assurez-vous que MCnet et YOLOP sont correctement définis et importés
checkpoint = torch.load('./weights/End-to-end.pth', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Ouvrir la caméra web (0 est l'indice de la caméra par défaut)
cap = cv2.VideoCapture(0)

# Vérifier si la caméra s'est ouverte correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

while True:
    # Lire une frame depuis la caméra
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionner la frame à la taille d'entrée du modèle (640x640)
    frame_resized = cv2.resize(frame, (640, 640))

    # Normaliser l'image (les pixels doivent être dans l'intervalle [0, 1])
    frame_resized = frame_resized / 255.0

    # Convertir l'image en tensor PyTorch et ajouter une dimension batch (1)
    frame_tensor = torch.tensor(frame_resized).float().unsqueeze(0).to(device)

    # Effectuer l'inférence avec le modèle
    det_out, da_seg, ll_seg = model(frame_tensor)

    # Afficher les résultats en tant que tensors pour chaque tâche
    print("Detection Output (Tensor):", det_out)
    print("Driving Area Segmentation (Tensor):", da_seg)
    print("Lane Line Segmentation (Tensor):", ll_seg)

    # Vous pouvez également les visualiser avec OpenCV ou effectuer des traitements supplémentaires

    # Convertir les sorties en format d'image pour affichage (optionnel)
    det_out_img = det_out.cpu().detach().numpy()  # Convertir en numpy pour OpenCV
    da_seg_img = da_seg.cpu().detach().numpy()
    ll_seg_img = ll_seg.cpu().detach().numpy()

    # Normalisation pour visualisation
    det_out_img = np.clip(det_out_img, 0, 1)
    da_seg_img = np.clip(da_seg_img, 0, 1)
    ll_seg_img = np.clip(ll_seg_img, 0, 1)

    # Affichage des images de sortie
    cv2.imshow('Detection Output', det_out_img[0, 0])  # Visualisation de la détection (exemple)
    cv2.imshow('Driving Area Segmentation', da_seg_img[0, 0])  # Visualisation de la zone praticable
    cv2.imshow('Lane Line Segmentation', ll_seg_img[0, 0])  # Visualisation des lignes de voie

    # Sortie de la boucle avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()

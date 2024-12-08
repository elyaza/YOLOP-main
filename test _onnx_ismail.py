import os
import cv2
import torch
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def infer_yolop(weight="yolop-640-640.onnx",
                img_path="./inference/images/7dd9ef45-f197db95.jpg"):

    ort.set_default_logger_severity(4)
    onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(onnx_path)
    print(f"Load {onnx_path} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)

    print("num outputs: ", len(outputs_info))

    save_ll_path = f"./inference/output/ll_onnx2.jpg"
    save_canny_path = f"./inference/output/canny_onnx2.jpg"
    save_hough_path = f"./inference/output/hough_onnx.jpg"
    save_merged_path = f"./inference/output/merged_onnx_jpg"

    img_bgr = cv2.imread(img_path)
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)  # (1, 3,640,640)

    # inference: (1,n,6) (1,2,640,640) (1,2,640,640)
    det_out, da_seg_out, ll_seg_out = ort_session.run(
        ['det_out', 'drive_area_seg', 'lane_line_seg'],
        input_feed={"images": img}
    )



    # select da & ll segment area.
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)

    # Convert lane line segmentation to black and white
    ll_seg_mask = ll_seg_mask * 255  # 255 for white, 0 for black
    ll_seg_mask = ll_seg_mask.astype(np.uint8)

    # Erode the blurred image
    kernel = np.ones((4, 4), np.uint8)
    erosion = cv2.erode(ll_seg_mask, kernel, iterations=1)

    

    ll_canny = cv2.Canny(erosion, 100, 200)  # Apply Canny with lower and upper thresholds



    # Optional: resize to original image size
    #ll_seg_mask_resized = cv2.resize(ll_seg_mask, (width, height), interpolation=cv2.INTER_LINEAR) 

    #ll_canny_resized = cv2.resize(ll_canny, (width, height), interpolation=cv2.INTER_LINEAR)

    lines = cv2.HoughLinesP( ll_canny , 1 , np.pi/180 ,threshold=50, minLineLength=50 , maxLineGap=50,)
    print(lines)
    ll_canny_bgr = cv2.cvtColor(ll_canny, cv2.COLOR_GRAY2BGR)
    ll_seg_mask_bgr = cv2.cvtColor(ll_seg_mask, cv2.COLOR_GRAY2BGR)

    pente_negative = []
    pente_positive = []

    for line in lines :
        x1 , y1 , x2 , y2 = line[0]
        m = (y2-y1)/(x2-x1)
        c = y1 - m*x1
        if m < 0 :
            pente_negative.append((m,c))
        else : pente_positive.append((m,c))
        x3 = 1000
        y3 = x3*m +c
        x4 = -1000
        y4 = x4*m + c
        cv2.line(ll_canny_bgr, (int(x4), int(y4)), (int(x3), int(y3)), (198,23,216),1,1 )
        cv2.line(ll_seg_mask_bgr, (int(x4), int(y4)), (int(x3), int(y3)), (198, 23, 216),1,1 )


    print(pente_positive)
    print(pente_negative)

    ## calcule du point d'intersection
    ## moyenne des pentes positives
    k,l = 0,0
    for i in pente_positive:
        k += i[0]
        l += i[1]
    mp = k / len(pente_positive)
    cp = l / len(pente_positive)

    ## moyenne des pentes positives
    k,l = 0,0
    for i in pente_negative:
        k += i[0]
        l += i[1]
    mn = k / len(pente_negative)
    cn = l / len(pente_negative)
    


    a = (cn - cp) / (mp - mn)
    b = mp*a + cp
    print((a,b))
    cv2.circle(ll_seg_mask_bgr, (int(a), int(b)), radius=20, color= (127,0,255) , thickness=2 )
    cv2.circle(ll_canny_bgr, (int(a), int(b)), radius=20, color= (127,0,255) , thickness=2 )
    cv2.circle(ll_canny_bgr, (0, 0), radius=20, color= (255,0,0) , thickness=2 )
    

    ## la droite de fuite
    hauteur, largeur = ll_canny.shape
    
    # Calculer les coordonnées du centre en bas
    centre_bas = (largeur // 2, hauteur - 1)  # y = hauteur - 1 pour le dernier pixel en bas

    cv2.line(ll_canny_bgr, (centre_bas[0], centre_bas[1]), (centre_bas[0], 1), (0,0,255),1,1 )
    cv2.line(
    ll_canny_bgr, 
    (int(centre_bas[0]), int(centre_bas[1])), 
    (int(a), int(b)), 
    (0, 0, 255), 
    1, 
    1
    )

        # Trouver les vecteurs directeurs
    u = (centre_bas[0] - a, centre_bas[1] - b)  # Vecteur de la première droite
    v = (0 , centre_bas[1] - 1)  # Vecteur de la deuxième droite

    # Produit scalaire
    produit_scalaire = u[0] * v[0] + u[1] * v[1]

    # Normes des vecteurs
    norme_u = np.sqrt(u[0]**2 + u[1]**2)
    norme_v = np.sqrt(v[0]**2 + v[1]**2)

    # Calcul du cosinus de l'angle
    cos_theta = produit_scalaire / (norme_u * norme_v)

    # S'assurer que le cosinus est dans l'intervalle valide [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calcul de l'angle en radians
    angle_radians = np.arccos(cos_theta)

    # Convertir en degrés si nécessaire
    angle_degres = np.degrees(angle_radians)
    
    print(angle_degres)





   
    cv2.imwrite(save_canny_path, ll_canny_bgr)
    cv2.imwrite(save_ll_path,ll_seg_mask_bgr )
    

    
    cv2.imshow("Lane Line Segmentation", ll_seg_mask_bgr)
    cv2.imshow("canny line segmentation",ll_canny_bgr)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="yolop-640-640.onnx")
    parser.add_argument('--img', type=str, default="./inference/images/22.jpg")
    args = parser.parse_args()
    infer_yolop(weight=args.weight, img_path=args.img)
# FastSAM-3D Realtime

Real-time 3D body keypoint extraction built on top of [SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body), running at **~15 FPS on an RTX 3090** with a 720p input stream.

---

## How we made it realtime

The original SAM-3D-Body pipeline runs at ~2–3 FPS out of the box. Here is every optimization applied to reach 15 FPS:

### 1. TensorRT backbone (biggest win)
The DINOv3 ViT-H backbone is converted to a TensorRT FP16 engine once, then reused at inference time.
- PyTorch fp32: ~350 ms/frame
- TensorRT FP16: ~30–40 ms/frame

```bash
python convert_backbone_tensorrt.py --all
```

### 2. TensorRT YOLO detector
The YOLO11 pose detector is also converted to a TRT FP16 engine.

```bash
python convert_yolo_pose_trt.py --model yolo11m-pose.pt --imgsz 640 --half
```

### 3. Body-only inference
Skipping the hand decoder saves ~100 ms per frame. For full-body motion capture without fine finger tracking this is a free win.

### 4. Fixed camera intrinsics
Running MoGe2 per frame to estimate the FOV costs ~80 ms. Estimating intrinsics once from a sample of frames and hardcoding them removes this entirely.

### 5. Skip intermediate predictions
Setting `INTERM_PRED_INTERVAL=999` makes the decoder only produce a prediction at the final layer instead of every layer, removing ~18 ms of MHR head calls.

### 6. No `torch.cuda.empty_cache()` per frame
The original estimator called `torch.cuda.empty_cache()` on every single frame. This forces a full CUDA synchronisation and memory sweep, costing 10–20 ms for no benefit — PyTorch's allocator handles memory reuse automatically.

### 7. Input resolution: film at 720p
The two hard resolution caps in the pipeline are:
- **YOLO detection**: internally runs at 640×640 regardless of input size
- **Backbone**: always sees a 512×512 crop of the detected person

Feeding 4K video costs **~57 ms just for frame decoding** (`cap.read()`). At 720p it drops to **~3 ms**. Accuracy is identical because the model never sees the extra pixels anyway.

| Resolution | Frame decode | Processing | Total overhead |
|---|---|---|---|
| 4K (3840×2160) | 57 ms | 1.2 ms | **58 ms** |
| 720p (1280×720) | 3 ms | 0.5 ms | **3.5 ms** |

---

## Setup

### Requirements
- CUDA 11.8+
- TensorRT 8.6+
- Python 3.10

### Install
```bash
git clone https://github.com/SoulieTechnologies/fastsam3d_realtime.git
cd fastsam3d_realtime
pip install -r requirements.txt
```

### Build TRT engines (run once)
```bash
# 1. Backbone
python convert_backbone_tensorrt.py --all

# 2. YOLO detector
python convert_yolo_pose_trt.py --model yolo11m-pose.pt --imgsz 640 --half
```

### Estimate camera intrinsics (run once per camera)
```bash
python -c "
import os, sys, cv2, numpy as np, torch
sys.path.insert(0, '.')
from tools.build_fov_estimator import FOVEstimator

fov = FOVEstimator(name='moge2', device='cuda')
cap = cv2.VideoCapture('your_video.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
results = []
for idx in np.linspace(0, total-1, 10, dtype=int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret: continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        K = fov.get_cam_intrinsics(rgb).squeeze().cpu().numpy()
    results.append([K[0,0], K[1,1], K[0,2], K[1,2]])
cap.release()
r = np.array(results)
print(f'--fx {r[:,0].mean():.1f} --fy {r[:,1].mean():.1f} --cx {r[:,2].mean():.1f} --cy {r[:,3].mean():.1f}')
"
```

---

## Demo

### Option A — Video file

```bash
# 1. Extract keypoints
python realtime_extractor.py \
  --source your_video.mp4 \
  --fx 664.7 --fy 664.7 --cx 640.0 --cy 360.0 \
  --headless

# 2. Visualize skeleton overlay
python visualize_skeleton_video.py \
  --npy ./output/joints_2d.npy \
  --video your_video.mp4 \
  --output ./output/skeleton_overlay.mp4
```

Then retrieve the result:
```bash
scp user@server:/path/to/Fast-SAM-3D-Body/output/skeleton_overlay.mp4 .
```

### Option B — Realtime webcam

```bash
python realtime_extractor.py \
  --source 0 \
  --fx 664.7 --fy 664.7 --cx 640.0 --cy 360.0 \
  --width 1280 --height 720
```

A live window shows the FPS counter. Press `q` to quit.

> **Tip:** Set your camera to 720p before running. Going above 720p gives no accuracy benefit (YOLO caps at 640px, backbone caps at 512px) and costs ~50 ms extra per frame in decode time.

---

## Output format

Both `joints_2d.npy` and `joints_3d.npy` are saved to `./output/` after each run.

| File | Shape | Description |
|---|---|---|
| `joints_2d.npy` | `(T, 70, 2)` | 2D pixel coordinates per frame |
| `joints_3d.npy` | `(T, 70, 3)` | 3D coordinates in camera space (metres) |

The 70 keypoints follow the [Goliath](https://github.com/facebookresearch/goliath) skeleton: 15 body joints, 5 foot joints, 21 keypoints per hand, and 4 extra body landmarks.

---

## Performance

Measured on RTX 3090, single person, body-only mode:

| Input | Median latency | FPS |
|---|---|---|
| 4K (no optimisations) | ~500 ms | ~2 |
| 4K (TRT backbone + YOLO) | ~100 ms | ~10 |
| 720p (all optimisations) | ~65 ms | **~15** |

---

## CLI reference

```
python realtime_extractor.py --help

  --source          Video path or webcam index (default: 0)
  --fx/fy/cx/cy     Camera intrinsics (use MoGe2 estimate if omitted)
  --detector_model  YOLO model path (auto-detects .engine if available)
  --body-only       Skip hand decoder (default: on)
  --headless        No display window
  --output_dir      Where to save .npy files (default: ./output)
  --gpu             GPU index
```

---

---

# FastSAM-3D Temps Réel

Extraction de keypoints 3D du corps en temps réel, basé sur [SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body), tournant à **~15 FPS sur RTX 3090** avec une entrée vidéo 720p.

---

## Comment on a rendu ça temps réel

Le pipeline SAM-3D-Body original tourne à ~2–3 FPS tel quel. Voici chaque optimisation appliquée pour atteindre 15 FPS :

### 1. Backbone TensorRT (gain le plus important)
Le backbone DINOv3 ViT-H est converti une fois en moteur TensorRT FP16, puis réutilisé à chaque inférence.
- PyTorch fp32 : ~350 ms/frame
- TensorRT FP16 : ~30–40 ms/frame

```bash
python convert_backbone_tensorrt.py --all
```

### 2. Détecteur YOLO en TensorRT
Le détecteur de pose YOLO11 est également converti en moteur TRT FP16.

```bash
python convert_yolo_pose_trt.py --model yolo11m-pose.pt --imgsz 640 --half
```

### 3. Inférence corps uniquement
Désactiver le décodeur des mains économise ~100 ms par frame. Pour de la capture de mouvement plein corps sans tracking fin des doigts, c'est un gain gratuit.

### 4. Intrinsèques caméra fixes
Faire tourner MoGe2 à chaque frame pour estimer le FOV coûte ~80 ms. Estimer les intrinsèques une fois sur un échantillon de frames et les fixer supprime ce coût entièrement.

### 5. Sauter les prédictions intermédiaires
Mettre `INTERM_PRED_INTERVAL=999` fait que le décodeur ne produit une prédiction qu'à la dernière couche plutôt qu'à chaque couche, supprimant ~18 ms d'appels à la tête MHR.

### 6. Supprimer `torch.cuda.empty_cache()` par frame
L'estimateur original appelait `torch.cuda.empty_cache()` à chaque frame. Cela force une synchronisation CUDA complète et un balayage mémoire, coûtant 10–20 ms sans aucun bénéfice — l'allocateur PyTorch gère la réutilisation mémoire automatiquement.

### 7. Résolution d'entrée : filmer en 720p
Les deux plafonds de résolution dans le pipeline sont :
- **Détection YOLO** : tourne en interne à 640×640 quelle que soit la taille de l'entrée
- **Backbone** : voit toujours un crop 512×512 de la personne détectée

Alimenter le pipeline avec une vidéo 4K coûte **~57 ms rien que pour le décodage** (`cap.read()`). En 720p ça tombe à **~3 ms**. La précision est identique car le modèle ne voit jamais les pixels supplémentaires.

| Résolution | Décodage frame | Traitement | Coût total |
|---|---|---|---|
| 4K (3840×2160) | 57 ms | 1.2 ms | **58 ms** |
| 720p (1280×720) | 3 ms | 0.5 ms | **3.5 ms** |

---

## Installation

### Prérequis
- CUDA 11.8+
- TensorRT 8.6+
- Python 3.10

### Installer
```bash
git clone https://github.com/SoulieTechnologies/fastsam3d_realtime.git
cd fastsam3d_realtime
pip install -r requirements.txt
```

### Construire les moteurs TRT (une seule fois)
```bash
# 1. Backbone
python convert_backbone_tensorrt.py --all

# 2. Détecteur YOLO
python convert_yolo_pose_trt.py --model yolo11m-pose.pt --imgsz 640 --half
```

### Estimer les intrinsèques caméra (une fois par caméra)
```bash
python -c "
import os, sys, cv2, numpy as np, torch
sys.path.insert(0, '.')
from tools.build_fov_estimator import FOVEstimator

fov = FOVEstimator(name='moge2', device='cuda')
cap = cv2.VideoCapture('votre_video.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
results = []
for idx in np.linspace(0, total-1, 10, dtype=int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret: continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        K = fov.get_cam_intrinsics(rgb).squeeze().cpu().numpy()
    results.append([K[0,0], K[1,1], K[0,2], K[1,2]])
cap.release()
r = np.array(results)
print(f'--fx {r[:,0].mean():.1f} --fy {r[:,1].mean():.1f} --cx {r[:,2].mean():.1f} --cy {r[:,3].mean():.1f}')
"
```

---

## Démo

### Option A — Fichier vidéo

```bash
# 1. Extraire les keypoints
python realtime_extractor.py \
  --source votre_video.mp4 \
  --fx 664.7 --fy 664.7 --cx 640.0 --cy 360.0 \
  --headless

# 2. Visualiser le squelette
python visualize_skeleton_video.py \
  --npy ./output/joints_2d.npy \
  --video votre_video.mp4 \
  --output ./output/skeleton_overlay.mp4
```

Récupérer le résultat en local :
```bash
scp user@serveur:/chemin/vers/Fast-SAM-3D-Body/output/skeleton_overlay.mp4 .
```

### Option B — Webcam temps réel

```bash
python realtime_extractor.py \
  --source 0 \
  --fx 664.7 --fy 664.7 --cx 640.0 --cy 360.0 \
  --width 1280 --height 720
```

Une fenêtre live affiche le compteur FPS. Appuyer sur `q` pour quitter.

> **Conseil :** Régler la caméra en 720p avant de lancer. Dépasser 720p n'apporte aucun gain de précision (YOLO plafonne à 640px, le backbone à 512px) et coûte ~50 ms de décodage supplémentaire par frame.

---

## Format de sortie

`joints_2d.npy` et `joints_3d.npy` sont sauvegardés dans `./output/` après chaque exécution.

| Fichier | Forme | Description |
|---|---|---|
| `joints_2d.npy` | `(T, 70, 2)` | Coordonnées 2D en pixels par frame |
| `joints_3d.npy` | `(T, 70, 3)` | Coordonnées 3D dans l'espace caméra (mètres) |

Les 70 keypoints suivent le squelette [Goliath](https://github.com/facebookresearch/goliath) : 15 articulations du corps, 5 articulations des pieds, 21 keypoints par main, et 4 repères corporels supplémentaires.

---

## Performances

Mesurées sur RTX 3090, une personne, mode corps uniquement :

| Entrée | Latence médiane | FPS |
|---|---|---|
| 4K (sans optimisations) | ~500 ms | ~2 |
| 4K (backbone TRT + YOLO) | ~100 ms | ~10 |
| 720p (toutes optimisations) | ~65 ms | **~15** |

---

## Référence CLI

```
python realtime_extractor.py --help

  --source          Chemin vidéo ou index webcam (défaut : 0)
  --fx/fy/cx/cy     Intrinsèques caméra (MoGe2 utilisé si omis)
  --detector_model  Chemin modèle YOLO (détecte .engine automatiquement)
  --body-only       Désactiver le décodeur mains (défaut : activé)
  --headless        Pas de fenêtre d'affichage
  --output_dir      Dossier de sauvegarde des .npy (défaut : ./output)
  --gpu             Index GPU
```

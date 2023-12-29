import json
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Importez les modules YOLOv8 et Ultralytics nécessaires
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Fonction pour extraire les informations des images à partir d'un manifest IIIF
def extract_image_info(manifest_path, output_csv):
    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)
        
        canvases = manifest_data['sequences'][0]['canvases']
        image_info_list = []

        for canvas in canvases:
            image_url = canvas['images'][0]['resource']['default']['@id']
            canvas_label = canvas.get('label', '')
            image_info_list.append({"image_url": image_url, "canvas_label": canvas_label})

        # Écrire les informations dans le fichier CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['image_url', 'canvas_label']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Écrire l'en-tête du CSV
            writer.writeheader()

            # Écrire les informations pour chaque image
            for image_info in image_info_list:
                writer.writerow(image_info)

    return image_info_list

# Chargez les informations des images à partir d'un manifest IIIF dans un fichier CSV
manifest_path = 'manifest.json'  # Mettez à jour avec le chemin correct
output_csv = 'cesr-multimedia-project.csv'  # Mettez à jour avec le nom correct
image_info_list = extract_image_info(manifest_path, output_csv)

# Chargez le modèle YOLOv8 pré-entraîné
weights = 'yolov8s.pt'  # Mettez à jour avec le chemin de votre modèle
device = select_device('')
model = attempt_load(weights, map_location=device)
img_size = check_img_size(640, s=model.stride.max())  # Ajustez la taille de l'image selon votre modèle

# Définissez le seuil de confiance
conf_thres = 0.5

# Dossier contenant vos images
img_folder = 'images'

# Créez et ouvrez le fichier CSV pour enregistrer les résultats
csv_result_file = 'resultats.csv'
with open(csv_result_file, 'w', newline='', encoding='utf-8') as csvfile:
    # Utilisez le module csv pour écrire dans le fichier CSV
    writer = csv.writer(csvfile)

    # Enregistrez l'en-tête avec les noms de catégories d'objets
    header = ['image_url'] + [f'category_{i}' for i in range(len(categories))]
    writer.writerow(header)

    # Parcourez toutes les images dans le dossier
    for img_path in tqdm(Path(img_folder).glob('*.jpg')):
        # Chargez l'image
        img0 = Image.open(img_path).convert('RGB')

        # Effectuez la détection d'objets
        img = LoadImages(img_path, img_size=img_size, stride=model.stride.max(), auto=False)[0]
        pred = model(img)

        # Appliquez la suppression des non-maximum
        pred = non_max_suppression(pred, conf_thres=conf_thres)

        # Récupérez les catégories d'objets détectées
        categories = [p[:, -1].tolist() for p in pred]

        # Enregistrez les résultats dans le fichier CSV
        row_data = [str(image_info["image_url"])] + [str(category) for category in categories[0]]
        writer.writerow(row_data)

# Assurez-vous de documenter votre script, en particulier le seuil de confiance et tout autre paramètre pertinent.

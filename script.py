import json
import csv

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

# Exemple d'utilisation
manifest_path = 'manifest.json'  # Assurez-vous que le chemin est correct
output_csv = 'cesr-multimedia-project.csv'  # Spécifiez le nom de votre fichier de sortie
image_info_list = extract_image_info(manifest_path, output_csv)
print(image_info_list)

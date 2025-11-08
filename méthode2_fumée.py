# -*- coding: utf-8 -*-
"""
Titre : Détection de panache de fumée par segmentation couleur
sur une image JPG en utilisant OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

repertoire_actuel = os.getcwd()

print("---------------------------------------------------------")
print(f"Python travaille actuellement dans le dossier :")
print(repertoire_actuel)
print("---------------------------------------------------------")

# ... la suite de votre code ...
# import cv2
# ...
# --- 1. Chargement de l'image ---
# Assurez-vous que l'image 'OUI.jpg' est dans le même dossier
try:
    image = cv2.imread('OUI.jpg')
    # OpenCV charge en BGR, on convertit en RGB pour Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
except:
    print("Erreur: Impossible de charger l'image 'OUI.jpg'.")
    print("Vérifiez que le fichier est bien dans le même répertoire que le script.")
    exit()

# --- 2. Conversion en HSV ---
# L'espace HSV est meilleur pour la segmentation par couleur
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# --- 3. Définition des plages de couleur pour la fumée ---
# C'est l'étape la plus délicate, elle demande des ajustements.
# La fumée est beige/marron/grise.
# Teinte (Hue) : 0-30 (couvre les marrons/oranges/jaunes)
# Saturation : 10-150 (ni totalement blanc/gris, ni totalement coloré)
# Valeur (Value) : 100-220 (ni noir, ni blanc brillant)

"""
# Plage inférieure
lower_smoke = np.array([10, 10, 100])
# Plage supérieure
upper_smoke = np.array([30, 150, 220])
"""

# Fumée blanche ou gris clair
lower_white_smoke = np.array([0, 0, 180])     # très peu saturée, assez lumineuse
upper_white_smoke = np.array([180, 40, 255])  # toutes teintes, faible saturation, très claire

# --- 4. Création du masque ---
mask =cv2.inRange(hsv, lower_white_smoke, upper_white_smoke)

# --- 5. Nettoyage du masque (optionnel mais recommandé) ---
# Élimine les petits points de bruit
n=5
kernel = np.ones((n, n), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# Épaissit un peu la détection
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

# --- 6. Trouver les contours (le "détourage") ---
# findContours modifie l'image source, d'où la copie
contours, _ = cv2.findContours(mask_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- 7. Dessiner les contours sur l'image originale ---
# On crée une copie de l'image RGB pour y dessiner
image_with_contours = image_rgb.copy()

# On dessine tous les contours trouvés en vert (0, 255, 0) avec une épaisseur de 3
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)

print(f"Trouvé {len(contours)} contour(s) de fumée.")

# --- 8. Affichage des résultats ---
plt.figure(figsize=(15, 7))

# Image originale
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Image Originale (OUI.jpg)')
plt.axis('off')

# Le masque (ce que l'IA "voit")
plt.subplot(1, 3, 2)
plt.imshow(mask_cleaned, cmap='gray')
plt.title('Masque de la Fumée')
plt.axis('off')

# Résultat final
plt.subplot(1, 3, 3)
plt.imshow(image_with_contours)
plt.title('Fumée Détourée')
plt.axis('off')

plt.tight_layout()
plt.show()
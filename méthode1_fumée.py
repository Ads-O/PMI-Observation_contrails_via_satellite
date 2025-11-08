# -*- coding: utf-8 -*-
"""
Titre : Simulation de composite "Ash RGB"
Basé sur la recette EUMETSAT pour la détection des cendres.
"""

import numpy as np
import matplotlib.pyplot as plt

#%%


def normalize(channel, vmin, vmax):
    channel = channel.astype(np.float32)
    vmin = float(vmin)
    vmax = float(vmax)
    
    norm = (channel - vmin) / (vmax - vmin) # normalisation 0-1
    norm = np.clip(norm, 0, 1) #valeurs limitées entre 0 et 1
    return (norm * 255).astype(np.uint8) # passage 0-255

# --- 1. DONNÉES SIMULÉES (Tableaux 2D de Températures en Kelvin) ---
N = 200

# Température de fond (ex: surface mer/terre à 290K)
IR10_8 = np.full((N, N), 290.0)
IR12_0 = np.full((N, N), 290.0)
IR8_7 = np.full((N, N), 290.0)

# Ajout d'un nuage de glace (Cirrus) froid (230K)
# Propriété : T(12.0) < T(10.8)
ice_slice = (slice(50, 100), slice(50, 100))
IR10_8[ice_slice] = 230.0
IR12_0[ice_slice] = 227.0 
IR8_7[ice_slice] = 230.5

# Ajout d'un panache de CENDRES (270K)
# Propriété : T(8.7) << T(10.8)
ash_slice = (slice(120, 170), slice(120, 170))
IR10_8[ash_slice] = 270.0
IR12_0[ash_slice] = 270.5 # Proche de T(10.8) -> Rouge faible
IR8_7[ash_slice] = 260.0  # Bien plus bas que T(10.8) -> Vert fort

# --- 2. Calcul des composantes RGB (selon la recette Ash RGB) ---
red_diff = IR12_0 - IR10_8
green_diff = IR10_8 - IR8_7
blue_abs = IR10_8

# --- 3. Normalisation (selon les plages EUMeTrain "Ash RGB") ---
# Plage Rouge: -4K à +2K
R = normalize(red_diff, -4, 2)
# Plage Verte: -4K à +5K
G = normalize(green_diff, -4, 5) 
# Plage Bleue: -30°C à +30°C (243.15K à 303.15K)
B = normalize(blue_abs, 243.15, 303.15)

# --- 4. Fusion en image RGB ---
RGB = np.stack([R, G, B], axis=2) 

# --- 5. Affichage ---
plt.figure(figsize=(20,20))
plt.imshow(RGB)
plt.title("Composite Ash RGB simulé")
plt.text(75, 75, 'Nuage de glace (Noir/Bleu)', color='white', ha='center')
plt.text(145, 145, 'Cendres (Jaune/Rouge)', color='black', ha='center')
plt.text(10, 10, 'Surface (Bleu/Vert)', color='white', ha='center')
plt.axis('off')
plt.show()
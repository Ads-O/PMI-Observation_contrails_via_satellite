import xml.etree.ElementTree as ET #pour lire les fichiers .XML
import xarray as xr #pour lire les fichiers .NC
import os
import numpy as np
import matplotlib.pyplot as plt

#%% #récupèrer les fichiers

#où se trouvent tous les fichiers sur mon pc (juste pour 13h)
dossier = r"C:\Users\Louis\OneDrive\Documents\IPSA\Aero_5\PMI\Images_satellites_8_avril\13h"

#analyse du fichier Manifest.xml
tree = ET.parse(os.path.join(dossier, "manifest.xml"))
root = tree.getroot()
ns = {'eum': 'http://www.eumetsat.int/sip'}

#on récupère tous les fichiers .nc
paths = [elem.text for elem in root.findall('.//eum:path', ns) if elem.text.endswith('.nc')]
#vérification du nombre de fichiers .nc récupérés
print("Nombre de fichiers trouvés :", len(paths))

#%% #test sur le premier fichier .nc

#on lit les variables
fichier_test = os.path.join(dossier, paths[0])
ds = xr.open_dataset(fichier_test)
print(ds)
print(ds.data_vars)

#on récupère IR10.5
ds_ir105 = xr.open_dataset(fichier_test, group="data/ir_105_hr/measured")
print(ds_ir105.data_vars) 
#récupération des variables utiles
eff = ds_ir105["effective_radiance"]
k_rad = ds_ir105["radiance_unit_conversion_coefficient"]
wn  = ds_ir105["radiance_to_bt_conversion_coefficient_wavenumber"]
c1  = ds_ir105["radiance_to_bt_conversion_constant_c1"]
c2  = ds_ir105["radiance_to_bt_conversion_constant_c2"]
a   = ds_ir105["radiance_to_bt_conversion_coefficient_a"]
b   = ds_ir105["radiance_to_bt_conversion_coefficient_b"]

#passage de la radiance effective à la radiance "physique"
L = eff * k_rad

#calcul de la température de brillance IR10.5 en K
BT_planck = c2 * wn / np.log(1 + (c1 * wn**3) / L)
BT_105 = a * BT_planck + b
BT_105.name = "BT_105"
print("BT 10.5 µm (K) :", float(BT_105.min()), "à", float(BT_105.max()))

#%% #pour tous les fichiers .nc

bt_105_list = []
for p in paths:
    fichier = os.path.join(dossier, p)
    #on ouvre le groupe IR10.5 mesuré
    ds_ir105_all = xr.open_dataset(fichier, group="data/ir_105_hr/measured")
    print(f"\nFichier : {p}")
    print("Variables dispo :", list(ds_ir105_all.data_vars))
    #si pas de radiance mesurée, on skip
    if "effective_radiance" not in ds_ir105_all:
        print("pas de 'effective_radiance' dans ce fichier (LUT / calibration), on le saute")
        ds_ir105_all.close()
        continue

    #récup des variables utiles
    eff_all = ds_ir105_all["effective_radiance"]
    k_rad_all = ds_ir105_all["radiance_unit_conversion_coefficient"]
    wn_all  = ds_ir105_all["radiance_to_bt_conversion_coefficient_wavenumber"]
    c1_all  = ds_ir105_all["radiance_to_bt_conversion_constant_c1"]
    c2_all  = ds_ir105_all["radiance_to_bt_conversion_constant_c2"]
    a_all   = ds_ir105_all["radiance_to_bt_conversion_coefficient_a"]
    b_all   = ds_ir105_all["radiance_to_bt_conversion_coefficient_b"]

    #passage radiance effective vers radiance "physique"
    L_all = eff_all * k_rad_all
    
    #BT planck + correction linéaire
    BT_planck_all = c2_all * wn_all / np.log(1 + (c1_all * wn_all**3) / L_all)
    BT_105_all = a_all * BT_planck_all + b_all
    BT_105_all.name = "BT_105"
    bt_105_list.append(BT_105_all)
    print(f"{p} -> BT 10.5 µm (K) : {float(BT_105_all.min()):.2f} à {float(BT_105_all.max()):.2f}")
    ds_ir105_all.close()

print("Nombre total de fichiers .nc traités pour IR10.5 :", len(bt_105_list))


#%% #combiner tous les fichiers .nc

#on convertit chaque DataArray en Dataset pour que combine_by_coords marche
bt_105_ds_list = [da.to_dataset() for da in bt_105_list]

#on combine par coordonnées (x, y)
bt_105_full_ds = xr.combine_by_coords(bt_105_ds_list)

#on récupère la DataArray
bt_105_full = bt_105_full_ds["BT_105"]

print(bt_105_full)


#%% #affichage

plt.figure(figsize=(10,8))
bt_105_full.plot.imshow()
plt.xlabel("")  
plt.ylabel("")
plt.title("BT 10.5 µm (K)")
plt.show()




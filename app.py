import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.segmentation import clear_border
import os
import io
from PIL import Image
import tempfile
import zipfile
import base64

def make_square(image, fill_color=(255, 255, 255)):
    """
    Rend l'image carrée en ajoutant des bordures blanches
    """
    height, width = image.shape[:2]
    
    # Déterminer la taille du carré (côté le plus grand)
    max_side = max(height, width)
    
    # Calculer les bordures à ajouter
    top = (max_side - height) // 2
    bottom = max_side - height - top
    left = (max_side - width) // 2
    right = max_side - width - left
    
    # Ajouter les bordures pour rendre l'image carrée
    square_image = cv2.copyMakeBorder(
        image, 
        top, 
        bottom, 
        left, 
        right, 
        cv2.BORDER_CONSTANT, 
        value=fill_color
    )
    
    return square_image

def process_image(image, params, expected_insects):
    """
    Traite une image selon les paramètres fournis et retourne les résultats
    """
    # Extraire les paramètres
    blur_kernel = params["blur_kernel"]
    adapt_block_size = params["adapt_block_size"]
    adapt_c = params["adapt_c"]
    morph_kernel = params["morph_kernel"]
    morph_iterations = params["morph_iterations"]
    min_area = params["min_area"]
    margin = params["margin"]
    auto_adjust = params["auto_adjust"]
    use_circularity = params.get("use_circularity", False)
    min_circularity = params.get("min_circularity", 0.3)
    
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien
    if blur_kernel > 1:
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    else:
        blurred = gray
    
    # Si auto-ajustement est activé
    if auto_adjust:
        # Plages de paramètres à explorer
        adapt_c_values = [-5, 0, 2, 5, 8, 10, 15]
        min_area_values = [20, 30, 50, 75, 100, 150, 200, 300]
        
        # Variables pour stocker les meilleurs paramètres
        best_params = {"adapt_c": 5, "min_area": 50}
        best_count_diff = float('inf')
        best_filtered_props = []
        
        # Tester toutes les combinaisons de paramètres
        for ac in adapt_c_values:
            for ma in min_area_values:
                # Appliquer le seuillage avec les paramètres actuels
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, adapt_block_size, ac
                )
                
                # Opérations morphologiques
                kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
                
                # Supprimer les objets qui touchent les bords
                cleared = clear_border(opening)
                
                # Étiqueter les composants connectés
                labels = measure.label(cleared)
                
                # Obtenir les propriétés des régions
                props = measure.regionprops(labels)
                
                # Filtrer les petites régions
                current_filtered_props = [prop for prop in props if prop.area >= ma]
                
                # Calculer la différence avec le nombre attendu
                count_diff = abs(len(current_filtered_props) - expected_insects)
                
                # Si cette combinaison donne un résultat plus proche du nombre attendu
                if count_diff < best_count_diff:
                    best_count_diff = count_diff
                    best_params["adapt_c"] = ac
                    best_params["min_area"] = ma
                    best_filtered_props = current_filtered_props
        
        # Utiliser les meilleurs paramètres trouvés
        adapt_c = best_params["adapt_c"]
        min_area = best_params["min_area"]
        
        # Recalculer une dernière fois avec les meilleurs paramètres
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c
        )
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        cleared = clear_border(opening)
        labels = measure.label(cleared)
        
        # Utiliser les propriétés filtrées optimales
        filtered_props = best_filtered_props
        
    else:
        # Traitement standard avec les paramètres choisis
        # Seuillage adaptatif
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c
        )

        # Étape de dilatation préliminaire pour connecter les structures fines
        connect_kernel = np.ones((5, 5), np.uint8)
        dilated_thresh = cv2.dilate(thresh, connect_kernel, iterations=2)
        
        # Puis procéder avec une fermeture morphologique plus agressive
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        closing = cv2.morphologyEx(dilated_thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        
        # Ensuite procéder avec l'ouverture comme avant
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)

        # Supprimer les objets qui touchent les bords
        cleared = clear_border(opening)

        # Étiqueter les composants connectés
        labels = measure.label(cleared)

        # Obtenir les propriétés des régions
        props = measure.regionprops(labels)

        # Filtrer les petites régions et appliquer le filtre de circularité si activé
        if use_circularity:
            filtered_props = []
            for prop in props:
                if prop.area >= min_area:
                    # Calculer la circularité: 4π × aire / périmètre²
                    perimeter = prop.perimeter
                    if perimeter > 0:  # Éviter division par zéro
                        circularity = 4 * np.pi * prop.area / (perimeter * perimeter)
                        if circularity >= min_circularity:
                            filtered_props.append(prop)
        else:
            filtered_props = [prop for prop in props if prop.area >= min_area]
    
    return {
        "blurred": blurred,
        "thresh": thresh, 
        "opening": opening,
        "labels": labels,
        "filtered_props": filtered_props,
        "adapt_c": adapt_c,
        "min_area": min_area
    }

def extract_insects(image, filtered_props, margin):
    """
    Extrait les insectes détectés de l'image et les rend carrés
    """
    extracted_insects = []
    
    # Pour chaque insecte détecté
    for i, prop in enumerate(filtered_props):
        # Obtenir les coordonnées de la boîte englobante
        minr, minc, maxr, maxc = prop.bbox

        # Ajouter une marge
        minr = max(0, minr - margin)
        minc = max(0, minc - margin)
        maxr = min(image.shape[0], maxr + margin)
        maxc = min(image.shape[1], maxc + margin)

        # Extraire l'insecte avec sa boîte englobante depuis l'image originale
        insect_roi = image[minr:maxr, minc:maxc].copy()
        roi_height, roi_width = insect_roi.shape[:2]
        
        # Créer un masque initial pour l'insecte
        mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        
        # 1. Marquer les pixels de l'insecte dans le masque
        for coord in prop.coords:
            if minr <= coord[0] < maxr and minc <= coord[1] < maxc:
                mask[coord[0] - minr, coord[1] - minc] = 255
        
        # 2. Appliquer une fermeture morphologique large pour connecter les régions 
        kernel_close = np.ones((7, 7), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=5)
        
        # 3. Trouver TOUS les contours (pas seulement les externes)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Créer un masque pour le dessin des contours remplis
        filled_mask = np.zeros_like(mask)
        
        # 5. Dessiner tous les contours trouvés avec remplissage
        for contour in contours:
            if cv2.contourArea(contour) > 20:  # Ignorer les très petits contours (bruit)
                cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # 6. Appliquer une dilatation pour connecter les parties proches
        kernel_dilate = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(filled_mask, kernel_dilate, iterations=3)
        
        # 7. Appliquer une nouvelle fermeture morphologique pour combler les trous restants
        final_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel_close, iterations=4)
        
        # 8. Pour les grands trous qui persistent, utiliser un remplissage par inondation
        # Créer une copie du masque avec une bordure
        mask_with_border = cv2.copyMakeBorder(final_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        
        # Créer un masque pour le floodfill
        flood_mask = np.zeros((roi_height+4, roi_width+4), dtype=np.uint8)
        
        # Remplir depuis les bords pour marquer tout l'extérieur
        cv2.floodFill(mask_with_border, flood_mask, (0, 0), 128)
        
        # Inverser: tout ce qui n'a pas été atteint est à l'intérieur d'un trou
        holes = np.where((mask_with_border != 128) & (mask_with_border != 255), 255, 0).astype(np.uint8)
        holes = holes[1:-1, 1:-1]  # Enlever la bordure
        
        # Ajouter les trous identifiés au masque final
        complete_mask = cv2.bitwise_or(final_mask, holes)
        
        # Traitement final - dilater légèrement pour lisser les bords
        kernel_smooth = np.ones((3, 3), np.uint8)
        smooth_mask = cv2.dilate(complete_mask, kernel_smooth, iterations=1)
        
        # Convertir en masque 3 canaux pour l'appliquer à l'image couleur
        mask_3ch = cv2.cvtColor(smooth_mask, cv2.COLOR_GRAY2BGR)
        
        # Créer l'image avec fond blanc
        white_bg = np.ones_like(insect_roi) * 255
        insect_on_white = np.where(mask_3ch == 255, insect_roi, white_bg)
        
        # Rendre l'image carrée
        square_insect = make_square(insect_on_white)
        
        extracted_insects.append({
            "image": square_insect,
            "index": i
        })
    
    return extracted_insects

def main():
    st.title("Détection et isolation d'insectes")
    st.write("Cette application permet de détecter des insectes sur un fond clair et de les isoler individuellement.")

    # Onglets pour l'application
    tab1, tab2 = st.tabs(["Application", "Guide d'utilisation"])
    
    with tab1:
        # Charger les images
        uploaded_files = st.file_uploader("Choisissez une ou plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            # Créer une liste pour stocker toutes les images et leurs résultats
            all_results = []
            
            # Paramètres ajustables dans la barre latérale
            st.sidebar.header("Paramètres de détection")
            
            # Demander le nombre attendu d'insectes
            expected_insects = st.sidebar.number_input("Nombre d'insectes attendus", min_value=1, value=5, step=1)
            
            # Ajout de configurations prédéfinies
            presets = {
                "Par défaut": {
                    "blur_kernel": 7,
                    "adapt_block_size": 35,
                    "adapt_c": 5,
                    "morph_kernel": 1,
                    "morph_iterations": 3,
                    "min_area": 1000,
                    "margin": 17
                },
                "Grands insectes": {
                    "blur_kernel": 7,
                    "adapt_block_size": 35,
                    "adapt_c": 8,
                    "morph_kernel": 5,
                    "morph_iterations": 2,
                    "min_area": 300,
                    "margin": 15
                },
                "Petits insectes": {
                    "blur_kernel": 3,
                    "adapt_block_size": 15,
                    "adapt_c": 2,
                    "morph_kernel": 3,
                    "morph_iterations": 1,
                    "min_area": 30,
                    "margin": 5
                },
                "Haute précision": {
                    "blur_kernel": 5,
                    "adapt_block_size": 25,
                    "adapt_c": 12,
                    "morph_kernel": 5,
                    "morph_iterations": 3,
                    "min_area": 150,
                    "margin": 10
                },
                "Arthropodes à pattes fines": {
                    "blur_kernel": 3,
                    "adapt_block_size": 21,
                    "adapt_c": 3,
                    "morph_kernel": 3,
                    "morph_iterations": 2,
                    "min_area": 150,
                    "margin": 20
                }
            }
            
            preset_choice = st.sidebar.selectbox(
                "Configurations prédéfinies", 
                ["Personnalisé", "Auto-ajustement"] + list(presets.keys()),
                index=2  # "Par défaut" sélectionné par défaut
            )
            
            # Initialisation des paramètres par défaut
            blur_kernel = 7
            adapt_block_size = 35
            adapt_c = 5
            morph_kernel = 1
            morph_iterations = 3
            min_area = 1000
            margin = 17
            auto_adjust = False
            
            # Utiliser les valeurs des presets ou permettre l'ajustement manuel
            if preset_choice == "Auto-ajustement":
                st.sidebar.info(f"Les paramètres seront ajustés automatiquement pour détecter {expected_insects} insectes.")
                auto_adjust = True
                
                # Permettre d'ajuster certains paramètres de base même en mode auto-ajustement
                blur_kernel = st.sidebar.slider("Taille du noyau de flou gaussien", 1, 21, 7, step=2)
                adapt_block_size = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, 35, step=2)
                morph_kernel = st.sidebar.slider("Taille du noyau morphologique", 1, 9, 1, step=2)
                morph_iterations = st.sidebar.slider("Itérations morphologiques", 1, 5, 3)
                
            elif preset_choice != "Personnalisé":
                preset = presets[preset_choice]
                blur_kernel = st.sidebar.slider("Taille du noyau de flou gaussien", 1, 21, preset["blur_kernel"], step=2)
                adapt_block_size = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, preset["adapt_block_size"], step=2)
                adapt_c = st.sidebar.slider("Constante de seuillage adaptatif", -10, 30, preset["adapt_c"])
                morph_kernel = st.sidebar.slider("Taille du noyau morphologique", 1, 9, preset["morph_kernel"], step=2)
                morph_iterations = st.sidebar.slider("Itérations morphologiques", 1, 5, preset["morph_iterations"])
                min_area = st.sidebar.slider("Surface minimale (pixels)", 10, 1000, preset["min_area"])
                margin = st.sidebar.slider("Marge autour des insectes", 0, 50, preset["margin"])
            else:
                # Paramètres complètement personnalisables
                blur_kernel = st.sidebar.slider("Taille du noyau de flou gaussien", 1, 21, 7, step=2)
                adapt_block_size = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, 35, step=2)
                adapt_c = st.sidebar.slider("Constante de seuillage adaptatif", -10, 30, 5)
                morph_kernel = st.sidebar.slider("Taille du noyau morphologique", 1, 9, 1, step=2)
                morph_iterations = st.sidebar.slider("Itérations morphologiques", 1, 5, 3)
                min_area = st.sidebar.slider("Surface minimale (pixels)", 10, 1000, 1000)
                margin = st.sidebar.slider("Marge autour des insectes", 0, 50, 17)

            # Ajouter un filtre de circularité
            use_circularity = st.sidebar.checkbox("Filtrer par circularité", value=False)
            min_circularity = 0.3
            if use_circularity:
                min_circularity = st.sidebar.slider("Circularité minimale", 0.0, 1.0, 0.3)
                
            # Regrouper les paramètres dans un dictionnaire
            params = {
                "blur_kernel": blur_kernel,
                "adapt_block_size": adapt_block_size,
                "adapt_c": adapt_c,
                "morph_kernel": morph_kernel,
                "morph_iterations": morph_iterations,
                "min_area": min_area,
                "margin": margin,
                "auto_adjust": auto_adjust,
                "use_circularity": use_circularity,
                "min_circularity": min_circularity
            }
            
            # Traiter chaque image téléchargée
            for file_index, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"### Image {file_index + 1}: {uploaded_file.name}")
                
                # Lire le contenu du fichier
                file_bytes = uploaded_file.getvalue()
                
                # Convertir le fichier en image
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Afficher l'image originale
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Image originale - {uploaded_file.name}", use_column_width=True)

                # Traitement de l'image
                with st.spinner(f"Traitement de l'image {file_index + 1} en cours..."):
                    results = process_image(image, params, expected_insects)
                    
                    # Ajouter les résultats à la liste globale
                    all_results.append({
                        "filename": uploaded_file.name,
                        "image": image,
                        "results": results
                    })
                    
                    # Afficher les résultats pour cette image
                    blurred = results["blurred"]
                    thresh = results["thresh"]
                    opening = results["opening"]
                    labels = results["labels"]
                    filtered_props = results["filtered_props"]
                    adapt_c = results["adapt_c"]
                    min_area = results["min_area"]
                    
                    # Si auto-ajustement est activé, afficher les paramètres optimaux trouvés
                    if auto_adjust:
                        st.success(f"Paramètres optimaux trouvés: adapt_c={adapt_c}, min_area={min_area}")
                    
                    # Créer une visualisation des étapes
                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(blurred, caption="Image floutée", use_column_width=True)
                        st.image(thresh, caption="Après seuillage adaptatif", use_column_width=True)

                    with col2:
                        st.image(opening, caption="Après opérations morphologiques", use_column_width=True)

                        # Créer une image colorée des labels pour visualisation
                        label_display = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                        for i, prop in enumerate(filtered_props):
                            color = np.random.randint(0, 255, size=3)
                            for coord in prop.coords:
                                label_display[coord[0], coord[1]] = color

                        st.image(label_display, caption=f"Insectes détectés: {len(filtered_props)}", use_column_width=True)

                    # Afficher des statistiques utiles
                    st.subheader(f"Statistiques de détection - {uploaded_file.name}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Nombre d'insectes", len(filtered_props))
                    col1.metric("Nombre attendu", expected_insects)
                    
                    if filtered_props:
                        areas = [prop.area for prop in filtered_props]
                        col2.metric("Surface moyenne (px)", f"{int(np.mean(areas))}")
                        col3.metric("Plage de tailles (px)", f"{int(min(areas))} - {int(max(areas))}")
                    
                    # Afficher l'écart par rapport au nombre attendu
                    diff = abs(len(filtered_props) - expected_insects)
                    if diff == 0:
                        st.success(f"✅ Nombre exact d'insectes détectés: {len(filtered_props)}")
                    elif diff <= 2:
                        st.warning(f"⚠️ {len(filtered_props)} insectes détectés (écart de {diff} par rapport au nombre attendu)")
                    else:
                        st.error(f"❌ {len(filtered_props)} insectes détectés (écart important de {diff} par rapport au nombre attendu)")
                
                    # Afficher quelques exemples d'insectes isolés (limités à 5 pour chaque image)
                    if filtered_props:
                        extracted_insects = extract_insects(image, filtered_props, margin)
                        
                        st.write(f"Aperçu des premiers insectes isolés de {uploaded_file.name}:")
                        preview_cols = st.columns(min(5, len(extracted_insects)))
                
                        for i, col in enumerate(preview_cols):
                            if i < len(extracted_insects):
                                insect = extracted_insects[i]
                                col.image(cv2.cvtColor(insect["image"], cv2.COLOR_BGR2RGB), caption=f"Insecte {i+1}", use_column_width=True)
                
                st.markdown("---")  # Séparateur entre les images
            
            # Option pour extraire et télécharger tous les insectes
            if st.button("Extraire et télécharger tous les insectes isolés (carrés)"):
                # Créer un dossier temporaire
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "insectes_isoles.zip")
            
                # Créer un fichier zip
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    # Pour chaque image traitée
                    for result_index, result in enumerate(all_results):
                        image = result["image"]
                        filename_base = os.path.splitext(result["filename"])[0]
                        filtered_props = result["results"]["filtered_props"]
                        
                        # Extraire les insectes de cette image
                        extracted_insects = extract_insects(image, filtered_props, margin)
                        
                        # Ajouter chaque insecte au zip
                        for insect in extracted_insects:
                            insect_img = insect["image"]
                            insect_index = insect["index"]
                            
                            # Nom de fichier avec l'image source et l'indice de l'insecte
                            temp_img_path = os.path.join(temp_dir, f"{filename_base}_insect_{insect_index+1}.jpg")
                            cv2.imwrite(temp_img_path, insect_img)
                            
                            # Ajouter au zip
                            zipf.write(temp_img_path, f"{filename_base}/insect_{insect_index+1}.jpg")
            
                # Créer un lien de téléchargement pour le zip
                with open(zip_path, "rb") as f:
                    bytes_data = f.read()
                    b64 = base64.b64encode(bytes_data).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="insectes_isoles.zip">Télécharger tous les insectes isolés (ZIP)</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    # Onglet pour le guide d'utilisation détaillé
    with tab2:
        st.header("Guide d'optimisation des paramètres")
        
        st.subheader("Configurations prédéfinies")
        st.write("""
        L'application propose plusieurs configurations prédéfinies pour différents types d'images:
        - **Par défaut**: Configuration optimisée basée sur les tests (flou gaussien: 7, bloc adaptatif: 35, seuillage: 5, noyau morphologique: 1, itérations: 3, surface min: 1000, marge: 17)
        - **Grands insectes**: Optimisée pour détecter des insectes de grande taille
        - **Petits insectes**: Optimisée pour les insectes de petite taille ou les détails fins
        - **Haute précision**: Réduit les fausses détections au prix d'une sensibilité légèrement plus faible
        - **Auto-ajustement**: Ajuste automatiquement les paramètres pour détecter le nombre d'insectes spécifié
        
        Vous pouvez commencer avec l'une de ces configurations puis ajuster les paramètres selon vos besoins.
        """)
        
        st.subheader("Traitement de plusieurs images")
        st.write("""
        L'application permet désormais de traiter plusieurs images simultanément:
        
        1. Téléchargez plusieurs images en les sélectionnant ensemble
        2. Chaque image sera traitée avec les mêmes paramètres
        3. Les insectes extraits seront regroupés par image source dans le fichier ZIP
        
        Cette fonctionnalité est particulièrement utile pour traiter des lots d'images similaires.
        """)
        
        st.subheader("Format des images extraites")
        st.write("""
        Les insectes extraits sont maintenant:
        
        1. **Rendus carrés** par ajout de bordures blanches autour de l'image originale
        2. **Fournis uniquement en format avec fond blanc** (plus d'option transparente)
        
        Ce format carré est idéal pour l'organisation et la présentation des résultats ou pour l'utilisation des images dans des applications de machine learning.
        """)
        
        st.subheader("Utilisation de l'auto-ajustement")
        st.write("""
        La fonctionnalité d'auto-ajustement permet de spécifier le nombre d'insectes attendus dans l'image:
        
        1. Indiquez le nombre d'insectes que vous savez présents dans l'image
        2. Sélectionnez le mode "Auto-ajustement" dans les configurations prédéfinies
        3. L'application testera différentes combinaisons de paramètres pour trouver celle qui détecte au mieux le nombre souhaité
st.write("""
        L'application testera différentes combinaisons de paramètres pour trouver celle qui détecte au mieux le nombre souhaité d'insectes.
        
        Cette fonctionnalité est particulièrement utile lorsque vous connaissez à l'avance le nombre d'insectes présents.
        """)
        
        st.subheader("Astuce pour les meilleurs résultats")
        st.write("""
        Pour obtenir les meilleurs résultats:
        
        1. **Qualité des images**: Utilisez des images bien éclairées avec un contraste suffisant entre les insectes et le fond
        2. **Fond uniforme**: Les fonds clairs et uniformes donnent les meilleurs résultats
        3. **Filtrer par circularité**: Activez cette option pour éliminer les fausses détections allongées ou irrégulières
        4. **Ajustement itératif**: Affinez progressivement les paramètres en observant les résultats à chaque étape
        
        N'hésitez pas à expérimenter avec différentes combinaisons de paramètres pour optimiser la détection selon vos images spécifiques.
        """)
        
        st.subheader("Fonctionnalités avancées")
        st.write("""
        Le filtrage par circularité permet d'éliminer les objets de forme très irrégulière qui ne sont probablement pas des insectes:
        
        - Une circularité de 1.0 correspond à un cercle parfait
        - Les valeurs recommandées sont entre 0.3 et 0.7 selon les types d'insectes
        - Les valeurs plus basses incluront davantage de formes allongées
        
        Cette option est particulièrement utile pour distinguer les insectes des débris ou artefacts dans l'image.
        """)
        
        st.subheader("Résolution des problèmes courants")
        st.write("""
        **Problème**: Trop peu d'insectes détectés
        - Solution: Diminuez la valeur de surface minimale ou augmentez la constante de seuillage adaptatif
        
        **Problème**: Trop d'insectes détectés (faux positifs)
        - Solution: Augmentez la valeur de surface minimale ou activez le filtre de circularité
        
        **Problème**: Les insectes sont fragmentés (détectés comme plusieurs objets)
        - Solution: Augmentez la taille du noyau morphologique ou le nombre d'itérations
        
        **Problème**: Les insectes proches sont fusionnés
        - Solution: Diminuez la taille du noyau morphologique ou activez le filtre de circularité
        """)

# Exécuter l'application
if __name__ == "__main__":
    main()

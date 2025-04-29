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

def process_image(image, preset_choice, expected_insects, params, auto_adjust=False):
    """
    Process a single image with the given parameters
    Returns the processed results and filtered regions
    """
    # Extract parameters
    blur_kernel = params["blur_kernel"]
    adapt_block_size = params["adapt_block_size"]
    adapt_c = params["adapt_c"]
    morph_kernel = params["morph_kernel"]
    morph_iterations = params["morph_iterations"]
    min_area = params["min_area"]
    margin = params["margin"]
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
    
    # Créer une image colorée des labels pour visualisation
    label_display = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for i, prop in enumerate(filtered_props):
        color = np.random.randint(0, 255, size=3)
        for coord in prop.coords:
            label_display[coord[0], coord[1]] = color
            
    # Regrouper tous les résultats de traitement
    results = {
        "blurred": blurred,
        "thresh": thresh,
        "opening": opening,
        "label_display": label_display,
        "labels": labels,
        "filtered_props": filtered_props,
        "adapt_c": adapt_c,
        "min_area": min_area
    }
    
    return results

def extract_insects(image, filtered_props, margin, temp_dir, image_index=0):
    """
    Extract individual insects from the image and save them
    Returns paths to the saved images
    """
    saved_paths = []
    
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
        
        # Créer une image avec transparence
        insect_transparent = np.zeros((roi_height, roi_width, 4), dtype=np.uint8)
        insect_transparent[:, :, :3] = insect_roi
        insect_transparent[:, :, 3] = smooth_mask
        
        # Sauvegarder les deux versions
        temp_img_path_white = os.path.join(temp_dir, f"image_{image_index}_insect_{i+1}_white.jpg")
        cv2.imwrite(temp_img_path_white, insect_on_white)
        
        temp_img_path_transparent = os.path.join(temp_dir, f"image_{image_index}_insect_{i+1}_transparent.png")
        cv2.imwrite(temp_img_path_transparent, insect_transparent)
        
        saved_paths.append((temp_img_path_white, temp_img_path_transparent, insect_on_white))
    
    return saved_paths

def main():
    st.title("Détection et isolation d'insectes")
    st.write("Cette application permet de détecter des insectes sur un fond clair et de les isoler individuellement.")

    # Onglets pour l'application
    tab1, tab2 = st.tabs(["Application", "Guide d'utilisation"])
    
    with tab1:
        # Charger des images multiples
        uploaded_files = st.file_uploader("Choisissez une ou plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            # Paramètres ajustables qui seront appliqués à toutes les images
            st.sidebar.header("Paramètres de détection")
            
            # Demander le nombre attendu d'insectes
            expected_insects = st.sidebar.number_input("Nombre d'insectes attendus par image", min_value=1, value=5, step=1)
            
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
                index=2  # "Par défaut" sélectionné par défaut (index 2 après "Personnalisé" et "Auto-ajustement")
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
                st.sidebar.info(f"Les paramètres seront ajustés automatiquement pour détecter {expected_insects} insectes par image.")
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
            
            # Organiser les paramètres dans un dictionnaire
            params = {
                "blur_kernel": blur_kernel,
                "adapt_block_size": adapt_block_size,
                "adapt_c": adapt_c,
                "morph_kernel": morph_kernel,
                "morph_iterations": morph_iterations,
                "min_area": min_area,
                "margin": margin,
                "use_circularity": use_circularity,
                "min_circularity": min_circularity
            }
            
            # Options pour le mode batch
            st.sidebar.header("Options de traitement par lot")
            batch_mode = st.sidebar.radio("Mode de traitement", ["Une image à la fois", "Toutes les images en lot"])
            
            # Créer un dossier temporaire pour les résultats
            temp_dir = tempfile.mkdtemp()
            
            if batch_mode == "Toutes les images en lot":
                if st.button("Traiter toutes les images"):
                    with st.spinner("Traitement des images en cours..."):
                        # Traiter toutes les images
                        all_results = []
                        all_images = []
                        all_saved_paths = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            # Mettre à jour la barre de progression
                            progress_bar.progress((i + 1) / len(uploaded_files))
                            
                            # Lire le contenu du fichier
                            file_bytes = uploaded_file.getvalue()
                            
                            # Convertir le fichier en image
                            nparr = np.frombuffer(file_bytes, np.uint8)
                            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            all_images.append(image)
                            
                            # Traiter l'image
                            results = process_image(image, preset_choice, expected_insects, params, auto_adjust)
                            all_results.append(results)
                            
                            # Extraire les insectes
                            saved_paths = extract_insects(image, results["filtered_props"], margin, temp_dir, i)
                            all_saved_paths.extend(saved_paths)
                        
                        # Créer un fichier zip avec tous les résultats
                        zip_path = os.path.join(temp_dir, "tous_insectes_isoles.zip")
                        
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for i, paths_set in enumerate(all_saved_paths):
                                white_path, transparent_path, _ = paths_set
                                zipf.write(white_path, os.path.basename(white_path))
                                zipf.write(transparent_path, os.path.basename(transparent_path))
                        
                        # Créer un lien de téléchargement pour le zip
                        with open(zip_path, "rb") as f:
                            bytes_data = f.read()
                            b64 = base64.b64encode(bytes_data).decode()
                            href = f'<a href="data:application/zip;base64,{b64}" download="tous_insectes_isoles.zip">Télécharger tous les insectes isolés (ZIP)</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        
                        # Afficher les résultats pour chaque image
                        for i, (image, results) in enumerate(zip(all_images, all_results)):
                            with st.expander(f"Résultats pour l'image {i+1}"):
                                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Image {i+1} originale", use_column_width=True)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(results["label_display"], caption=f"Insectes détectés: {len(results['filtered_props'])}", use_column_width=True)
                                
                                with col2:
                                    st.metric("Nombre d'insectes", len(results["filtered_props"]))
                                    if auto_adjust:
                                        st.text(f"Paramètres optimaux trouvés: adapt_c={results['adapt_c']}, min_area={results['min_area']}")
                                
                                # Afficher les 5 premiers insectes isolés
                                if results["filtered_props"]:
                                    st.write("Aperçu des premiers insectes isolés:")
                                    preview_cols = st.columns(min(5, len(results["filtered_props"])))
                                    
                                    for j, col in enumerate(preview_cols):
                                        if j < len(results["filtered_props"]):
                                            # Récupérer l'image isolée correspondante
                                            matching_paths = [p for p in all_saved_paths if f"image_{i}_insect_{j+1}" in p[0]]
                                            if matching_paths:
                                                _, _, insect_image = matching_paths[0]
                                                col.image(cv2.cvtColor(insect_image, cv2.COLOR_BGR2RGB), caption=f"Insecte {j+1}", use_column_width=True)
            else:
                # Mode de traitement d'une image à la fois
                # Selection d'image avec un selectbox
                if len(uploaded_files) > 0:
                    selected_image_index = st.selectbox(
                        "Sélectionner une image à traiter",
                        range(len(uploaded_files)),
                        format_func=lambda i: f"Image {i+1}"
                    )
                    
                    # Lire le contenu du fichier sélectionné
                    selected_file = uploaded_files[selected_image_index]
                    file_bytes = selected_file.getvalue()
                    
                    # Convertir le fichier en image
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Afficher l'image originale
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image originale", use_column_width=True)
                    
                    # Traitement de l'image
                    if st.button("Traiter cette image"):
                        with st.spinner("Traitement de l'image en cours..."):
                            # Traiter l'image
                            results = process_image(image, preset_choice, expected_insects, params, auto_adjust)
                            
                            # Créer une visualisation des étapes
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(results["blurred"], caption="Image floutée", use_column_width=True)
                                st.image(results["thresh"], caption="Après seuillage adaptatif", use_column_width=True)
                            
                            with col2:
                                st.image(results["opening"], caption="Après opérations morphologiques", use_column_width=True)
                                st.image(results["label_display"], caption=f"Insectes détectés: {len(results['filtered_props'])}", use_column_width=True)
                            
                            # Afficher des statistiques utiles
                            st.subheader("Statistiques de détection")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Nombre d'insectes", len(results["filtered_props"]))
                            col1.metric("Nombre attendu", expected_insects)
                            
                            if results["filtered_props"]:
                                areas = [prop.area for prop in results["filtered_props"]]
                                col2.metric("Surface moyenne (px)", f"{int(np.mean(areas))}")
                                col3.metric("Plage de tailles (px)", f"{int(min(areas))} - {int(max(areas))}")
                            
                            # Afficher l'écart par rapport au nombre attendu
                            diff = abs(len(results["filtered_props"]) - expected_insects)
                            if diff == 0:
                                st.success(f"✅ Nombre exact d'insectes détectés: {len(results['filtered_props'])}")
                            elif diff <= 2:
                                st.warning(f"⚠️ {len(results['filtered_props'])} insectes détectés (écart de {diff} par rapport au nombre attendu)")
                            else:
                                st.error(f"❌ {len(results['filtered_props'])} insectes détectés (écart important de {diff} par rapport au nombre attendu)")
                                
                                # Suggérer l'auto-ajustement si on n'est pas déjà en mode auto
                                if not auto_adjust:
                                    if st.button("Essayer l'auto-ajustement"):
                                        st.session_state['auto_adjust'] = True
                                        st.session_state['preset_choice'] = "Auto-ajustement"
                                        st.experimental_rerun()
                            
                            # Option pour extraire et télécharger les insectes
                            if st.button("Extraire et télécharger les insectes isolés"):
                                # Extraire les insectes
                                saved_paths = extract_insects(image, results["filtered_props"], margin, temp_dir, selected_image_index)
                                
                                # Créer un fichier zip
                                zip_path = os.path.join(temp_dir, f"insectes_isoles_image_{selected_image_index+1}.zip")
                                
                                with zipfile.ZipFile(zip_path, 'w') as zipf:
                                    for white_path, transparent_path, _ in saved_paths:
                                        zipf.write(white_path, os.path.basename(white_path))
                                        zipf.write(transparent_path, os.path.basename(transparent_path))
                                
                                # Créer un lien de téléchargement pour le zip
                                with open(zip_path, "rb") as f:
                                    bytes_data = f.read()
                                    b64 = base64.b64encode(bytes_data).decode()
                                    href = f'<a href="data:application/zip;base64,{b64}" download="insectes_isoles_image_{selected_image_index+1}.zip">Télécharger les insectes isolés (ZIP)</a>'
                                    st.markdown(href, unsafe_allow_html=True)

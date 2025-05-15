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
    Rend lʼimage carrée en ajoutant des bordures blanches.
    """
    height, width = image.shape[:2]
    
    max_side = max(height, width)
    
    top = (max_side - height) // 2
    bottom = max_side - height - top
    left = (max_side - width) // 2
    right = max_side - width - left
    
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
    Traite une image selon les paramètres fournis et retourne les résultats.
    """
    blur_kernel = params["blur_kernel"]
    adapt_block_size = params["adapt_block_size"]
    adapt_c = params["adapt_c"]
    morph_kernel_size = params["morph_kernel"] # Renamed for clarity, as it's a size
    morph_iterations = params["morph_iterations"]
    min_area = params["min_area"]
    # margin is used in extract_insects, not here
    auto_adjust = params["auto_adjust"]
    use_circularity = params.get("use_circularity", False)
    min_circularity = params.get("min_circularity", 0.3)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if blur_kernel > 1:
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    else:
        blurred = gray
    
    if auto_adjust:
        adapt_c_values = [-5, 0, 2, 5, 8, 10, 15]
        min_area_values = [20, 30, 50, 75, 100, 150, 200, 300]
        
        best_params_auto = {"adapt_c": 5, "min_area": 50} # Changed variable name
        best_count_diff = float('inf')
        best_filtered_props = []
        
        for ac_auto in adapt_c_values: # Changed variable name
            for ma_auto in min_area_values: # Changed variable name
                thresh_auto = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, adapt_block_size, ac_auto
                )
                
                kernel_auto = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
                opening_auto = cv2.morphologyEx(thresh_auto, cv2.MORPH_OPEN, kernel_auto, iterations=morph_iterations)
                cleared_auto = clear_border(opening_auto)
                labels_auto = measure.label(cleared_auto)
                props_auto = measure.regionprops(labels_auto)
                
                current_filtered_props = [prop for prop in props_auto if prop.area >= ma_auto]
                count_diff = abs(len(current_filtered_props) - expected_insects)
                
                if count_diff < best_count_diff:
                    best_count_diff = count_diff
                    best_params_auto["adapt_c"] = ac_auto
                    best_params_auto["min_area"] = ma_auto
                    best_filtered_props = current_filtered_props
        
        adapt_c = best_params_auto["adapt_c"]
        min_area = best_params_auto["min_area"]
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c
        )
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        cleared = clear_border(opening)
        labels = measure.label(cleared)
        filtered_props = best_filtered_props
        
    else:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c
        )

        connect_kernel = np.ones((5, 5), np.uint8) # Dilate to connect fine structures
        dilated_thresh = cv2.dilate(thresh, connect_kernel, iterations=2)
        
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        # Using MORPH_CLOSE then MORPH_OPEN (Closing followed by Opening)
        # This is more robust for filling holes and then removing small noise
        closing = cv2.morphologyEx(dilated_thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1) # Iterations for opening is often 1

        cleared = clear_border(opening)
        labels = measure.label(cleared)
        props = measure.regionprops(labels)

        if use_circularity:
            filtered_props = []
            for prop in props:
                if prop.area >= min_area:
                    perimeter = prop.perimeter
                    if perimeter > 0:
                        circularity = 4 * np.pi * prop.area / (perimeter * perimeter)
                        if circularity >= min_circularity:
                            filtered_props.append(prop)
        else:
            filtered_props = [prop for prop in props if prop.area >= min_area]
    
    return {
        "blurred": blurred,
        "thresh": thresh, 
        "opening": opening, # This is the result of CLmorph_kernel_sizeOSE then OPEN
        "labels": labels,
        "filtered_props": filtered_props,
        "adapt_c": adapt_c, # Return the used adapt_c
        "min_area": min_area # Return the used min_area
    }

def extract_insects(image, filtered_props, margin):
    """
    Extrait les insectes détectés de lʼimage, les nettoie et les rend carrés avec fond blanc.
    """
    extracted_insects = []
    
    for i, prop in enumerate(filtered_props):
        minr, minc, maxr, maxc = prop.bbox
        minr = max(0, minr - margin)
        minc = max(0, minc - margin)
        maxr = min(image.shape[0], maxr + margin)
        maxc = min(image.shape[1], maxc + margin)

        insect_roi = image[minr:maxr, minc:maxc].copy()
        roi_height, roi_width = insect_roi.shape[:2]

        if roi_height == 0 or roi_width == 0: # Skip if ROI is empty
            continue
            
        # 1. Initial mask from prop.coords within the ROI
        mask_from_coords = np.zeros((roi_height, roi_width), dtype=np.uint8)
        for coord in prop.coords:
            r_roi, c_roi = coord[0] - minr, coord[1] - minc # Coords relative to ROI
            if 0 <= r_roi < roi_height and 0 <= c_roi < roi_width:
                mask_from_coords[r_roi, c_roi] = 255
        
        # 2. Morphological closing to connect initial mask parts
        kernel_close_initial = np.ones((5, 5), np.uint8) # Smaller kernel for initial connection
        mask_step1 = cv2.morphologyEx(mask_from_coords, cv2.MORPH_CLOSE, kernel_close_initial, iterations=2)

        # 3. Find contours and fill them to create a solid mask
        contours, _ = cv2.findContours(mask_step1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask_step2_filled = np.zeros_like(mask_step1)
        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 10: # Filter small noise contours
                    cv2.drawContours(mask_step2_filled, [contour], -1, 255, thickness=cv2.FILLED)
        
        # 4. Dilate slightly to ensure connected components, then close to fill holes
        kernel_dilate_connect = np.ones((3, 3), np.uint8) # Small dilation
        mask_step3_dilated = cv2.dilate(mask_step2_filled, kernel_dilate_connect, iterations=1)
        
        kernel_close_holes = np.ones((7, 7), np.uint8) # Larger kernel for hole filling
        mask_step4_closed = cv2.morphologyEx(mask_step3_dilated, cv2.MORPH_CLOSE, kernel_close_holes, iterations=3)

        # 5. Flood fill for any remaining large internal holes
        mask_bordered_for_floodfill = cv2.copyMakeBorder(mask_step4_closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        floodfill_aux_mask = np.zeros((roi_height + 4, roi_width + 4), dtype=np.uint8)
        cv2.floodFill(mask_bordered_for_floodfill, floodfill_aux_mask, (0, 0), 128) # Mark exterior
        
        holes_mask = np.where((mask_bordered_for_floodfill != 128) & (mask_bordered_for_floodfill != 255), 255, 0).astype(np.uint8)
        holes_mask = holes_mask[1:-1, 1:-1] # Crop back
        
        mask_step5_holes_filled = cv2.bitwise_or(mask_step4_closed, holes_mask)

        # 6. Refinement: Take the largest contour from the processed mask
        # This helps to remove any small, detached artifacts or "over-masking" at the periphery.
        contours_final_selection, _ = cv2.findContours(mask_step5_holes_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clean_mask = np.zeros_like(mask_step5_holes_filled) # Start with an empty mask
        if contours_final_selection:
            largest_contour = max(contours_final_selection, key=cv2.contourArea)
            # Ensure the largest contour has a reasonable area before drawing
            if cv2.contourArea(largest_contour) > 20: # Min area for a valid insect part
                 cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # 7. Final smoothing of the clean_mask edges
        kernel_smooth_final = np.ones((3, 3), np.uint8)
        final_smooth_mask = cv2.dilate(clean_mask, kernel_smooth_final, iterations=1) # Gentle dilation for smoothing

        # Apply the final mask
        mask_3ch = cv2.cvtColor(final_smooth_mask, cv2.COLOR_GRAY2BGR)
        white_bg = np.ones_like(insect_roi, dtype=np.uint8) * 255 # Ensure uint8
        
        insect_on_white = np.where(mask_3ch == 255, insect_roi, white_bg)
        
        # Make the extracted insect image square
        square_insect = make_square(insect_on_white, fill_color=(255, 255, 255))
        
        extracted_insects.append({
            "image": square_insect,
            "index": i
        })
    
    return extracted_insects

def main():
    st.title("Détection et isolation dʼinsectes")
    st.write("Cette application permet de détecter des insectes sur un fond clair et de les isoler individuellement.")

    tab1, tab2 = st.tabs(["Application", "Guide dʼutilisation"])
    
    with tab1:
        uploaded_files = st.file_uploader("Choisissez une ou plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            all_results_data = [] # Renamed for clarity
            
            st.sidebar.header("Paramètres de détection")
            expected_insects = st.sidebar.number_input("Nombre dʼinsectes attendus", min_value=1, value=5, step=1)
            
            presets = {
                "Par défaut": {
                    "blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 5,
                    "morph_kernel": 3, "morph_iterations": 2, # Adjusted default morph
                    "min_area": 100, "margin": 15 # Adjusted default area/margin
                },
                "Grands insectes": {
                    "blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 8,
                    "morph_kernel": 5, "morph_iterations": 2, "min_area": 300, "margin": 15
                },
                "Petits insectes": {
                    "blur_kernel": 3, "adapt_block_size": 15, "adapt_c": 2,
                    "morph_kernel": 3, "morph_iterations": 1, "min_area": 30, "margin": 5
                },
                "Haute précision": {
                    "blur_kernel": 5, "adapt_block_size": 25, "adapt_c": 12,
                    "morph_kernel": 3, "morph_iterations": 2, # Adjusted morph
                    "min_area": 150, "margin": 10
                },
                "Arthropodes à pattes fines": {
                    "blur_kernel": 3, "adapt_block_size": 21, "adapt_c": 3,
                    "morph_kernel": 3, "morph_iterations": 2, "min_area": 150, "margin": 20
                }
            }
            
            preset_choice = st.sidebar.selectbox(
                "Configurations prédéfinies", 
                ["Personnalisé", "Auto-ajustement"] + list(presets.keys()),
                index=2 
            )
            
            # Default params (will be overridden by preset or custom)
            params_config = presets["Par défaut"].copy()
            auto_adjust_mode = False # Renamed

            if preset_choice == "Auto-ajustement":
                st.sidebar.info(f"Ajustement auto. pour {expected_insects} insectes.")
                auto_adjust_mode = True
                # Base params for auto-adjustment (some are fixed, others (adapt_c, min_area) are auto-tuned)
                params_config["blur_kernel"] = st.sidebar.slider("Noyau de flou (Auto)", 1, 21, params_config["blur_kernel"], step=2)
                params_config["adapt_block_size"] = st.sidebar.slider("Bloc adaptatif (Auto)", 3, 51, params_config["adapt_block_size"], step=2)
                params_config["morph_kernel"] = st.sidebar.slider("Noyau morph. (Auto)", 1, 9, params_config["morph_kernel"], step=2)
                params_config["morph_iterations"] = st.sidebar.slider("Itérations morph. (Auto)", 1, 5, params_config["morph_iterations"])
                # adapt_c and min_area will be determined by auto-adjust logic
            elif preset_choice != "Personnalisé":
                params_config = presets[preset_choice].copy()
                # Allow override of preset values
                params_config["blur_kernel"] = st.sidebar.slider(f"Noyau de flou ({preset_choice})", 1, 21, params_config["blur_kernel"], step=2)
                params_config["adapt_block_size"] = st.sidebar.slider(f"Bloc adaptatif ({preset_choice})", 3, 51, params_config["adapt_block_size"], step=2)
                params_config["adapt_c"] = st.sidebar.slider(f"Constante seuillage ({preset_choice})", -10, 30, params_config["adapt_c"])
                params_config["morph_kernel"] = st.sidebar.slider(f"Noyau morph. ({preset_choice})", 1, 9, params_config["morph_kernel"], step=2)
                params_config["morph_iterations"] = st.sidebar.slider(f"Itérations morph. ({preset_choice})", 1, 5, params_config["morph_iterations"])
                params_config["min_area"] = st.sidebar.slider(f"Surface min. ({preset_choice})", 10, 1000, params_config["min_area"])
                params_config["margin"] = st.sidebar.slider(f"Marge ({preset_choice})", 0, 50, params_config["margin"])
            else: # Personnalisé
                params_config["blur_kernel"] = st.sidebar.slider("Noyau de flou", 1, 21, params_config["blur_kernel"], step=2)
                params_config["adapt_block_size"] = st.sidebar.slider("Bloc adaptatif", 3, 51, params_config["adapt_block_size"], step=2)
                params_config["adapt_c"] = st.sidebar.slider("Constante seuillage", -10, 30, params_config["adapt_c"])
                params_config["morph_kernel"] = st.sidebar.slider("Noyau morph.", 1, 9, params_config["morph_kernel"], step=2)
                params_config["morph_iterations"] = st.sidebar.slider("Itérations morph.", 1, 5, params_config["morph_iterations"])
                params_config["min_area"] = st.sidebar.slider("Surface min.", 10, 1000, params_config["min_area"])
                params_config["margin"] = st.sidebar.slider("Marge", 0, 50, params_config["margin"])

            params_config["use_circularity"] = st.sidebar.checkbox("Filtrer par circularité", value=False)
            if params_config["use_circularity"]:
                params_config["min_circularity"] = st.sidebar.slider("Circularité minimale", 0.0, 1.0, 0.3, step=0.05)
            else:
                params_config["min_circularity"] = 0.3 # Default, not used if checkbox is off
            
            params_config["auto_adjust"] = auto_adjust_mode # Add auto_adjust flag to params

            for file_index, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"### Image {file_index + 1}: {uploaded_file.name}")
                
                file_bytes = uploaded_file.getvalue()
                nparr = np.frombuffer(file_bytes, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Renamed to cv_image

                st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption=f"Image originale - {uploaded_file.name}", use_column_width=True)

                with st.spinner(f"Traitement de lʼimage {file_index + 1} en cours..."):
                    # Pass cv_image (original, not squared yet) to process_image
                    # Squaring happens to the extracted ROIs later
                    processed_data = process_image(cv_image, params_config, expected_insects) 
                    
                    all_results_data.append({
                        "filename": uploaded_file.name,
                        "image": cv_image, # Store original image for extraction
                        "results": processed_data
                    })
                    
                    blurred_img = processed_data["blurred"]
                    thresh_img = processed_data["thresh"]
                    opening_img = processed_data["opening"]
                    labels_img = processed_data["labels"]
                    current_filtered_props = processed_data["filtered_props"] # Renamed
                    # Get the actual adapt_c and min_area used (important for auto-adjust)
                    actual_adapt_c = processed_data["adapt_c"] 
                    actual_min_area = processed_data["min_area"]
                    
                    if auto_adjust_mode:
                        st.success(f"Paramètres optimaux trouvés: adapt_c={actual_adapt_c}, min_area={actual_min_area}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(blurred_img, caption="Image floutée", use_column_width=True)
                        st.image(thresh_img, caption="Après seuillage adaptatif", use_column_width=True)
                    with col2:
                        st.image(opening_img, caption="Après opérations morphologiques", use_column_width=True)
                        
                        label_display = np.zeros((labels_img.shape[0], labels_img.shape[1], 3), dtype=np.uint8)
                        for prop_item in current_filtered_props: # Renamed
                            color = np.random.randint(0, 255, size=3)
                            for coord in prop_item.coords:
                                label_display[coord[0], coord[1]] = color
                        st.image(label_display, caption=f"Insectes détectés: {len(current_filtered_props)}", use_column_width=True)

                    st.subheader(f"Statistiques de détection - {uploaded_file.name}")
                    stat_col1, stat_col2, stat_col3 = st.columns(3) # Renamed
                    stat_col1.metric("Nombre dʼinsectes", len(current_filtered_props))
                    stat_col1.metric("Nombre attendu", expected_insects)
                    
                    if current_filtered_props:
                        areas = [prop.area for prop in current_filtered_props]
                        stat_col2.metric("Surface moyenne (px)", f"{int(np.mean(areas)) if areas else 0}")
                        stat_col3.metric("Plage de tailles (px)", f"{int(min(areas)) if areas else 0} - {int(max(areas)) if areas else 0}")
                    
                    diff = abs(len(current_filtered_props) - expected_insects)
                    if diff == 0:
                        st.success(f"✅ Nombre exact dʼinsectes détectés: {len(current_filtered_props)}")
                    elif diff <= 2:
                        st.warning(f"⚠️ {len(current_filtered_props)} insectes détectés (écart de {diff})")
                    else:
                        st.error(f"❌ {len(current_filtered_props)} insectes détectés (écart de {diff})")
                
                    if current_filtered_props:
                        # Pass original image and current props for extraction
                        # The margin from params_config is used here
                        extracted_insects_list = extract_insects(cv_image, current_filtered_props, params_config["margin"])
                        
                        st.write(f"Aperçu des premiers insectes isolés de {uploaded_file.name}:")
                        preview_cols = st.columns(min(5, len(extracted_insects_list)))
                
                        for i_prev, col_prev in enumerate(preview_cols): # Renamed
                            if i_prev < len(extracted_insects_list):
                                insect_data = extracted_insects_list[i_prev] # Renamed
                                col_prev.image(cv2.cvtColor(insect_data["image"], cv2.COLOR_BGR2RGB), caption=f"Insecte {insect_data['index']+1}", use_column_width=True)
                
                st.markdown("---")
            
            if st.button("Extraire et télécharger tous les insectes isolés (carrés, fond blanc)"):
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "insectes_isoles.zip")
            
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for result_item in all_results_data: # Renamed
                        image_orig = result_item["image"] # Original (not squared) image
                        filename_base = os.path.splitext(result_item["filename"])[0]
                        props_for_extraction = result_item["results"]["filtered_props"]
                        
                        # Use the margin from the current global params_config for all extractions
                        extracted_insects_for_zip = extract_insects(image_orig, props_for_extraction, params_config["margin"])
                        
                        for insect_detail in extracted_insects_for_zip: # Renamed
                            insect_img_square = insect_detail["image"] # This is already squared with white bg
                            insect_idx = insect_detail["index"]
                            
                            # Save as JPG (no transparency)
                            temp_img_filename = f"{filename_base}_insect_{insect_idx+1}.jpg"
                            temp_img_path = os.path.join(temp_dir, temp_img_filename)
                            cv2.imwrite(temp_img_path, insect_img_square)
                            
                            zipf.write(temp_img_path, os.path.join(filename_base, temp_img_filename)) # Store in subfolder per image
            
                with open(zip_path, "rb") as f:
                    bytes_data = f.read()
                    b64 = base64.b64encode(bytes_data).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="insectes_isoles.zip">Télécharger tous les insectes isolés (ZIP)</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    with tab2:
        st.header("Guide dʼoptimisation des paramètres")
        
        st.subheader("Configurations prédéfinies")
        st.write("""
        Lʼapplication propose plusieurs configurations prédéfinies pour différents types dʼimages:
        - **Par défaut**: Configuration équilibrée (flou: 7, bloc adaptatif: 35, const. seuil: 5, noyau morph: 3, itérations morph: 2, surface min: 100, marge: 15).
        - **Grands insectes**: Optimisée pour détecter des insectes de grande taille.
        - **Petits insectes**: Optimisée pour les insectes de petite taille ou les détails fins.
        - **Haute précision**: Vise à réduire les fausses détections.
        - **Arthropodes à pattes fines**: Configuration spécifique pour les insectes avec des appendices fins.
        - **Auto-ajustement**: Ajuste automatiquement `Constante de seuillage adaptatif` et `Surface minimale` pour détecter le nombre dʼinsectes spécifié. Les autres paramètres (flou, bloc, morphologie) peuvent toujours être réglés manuellement.
        
        Vous pouvez commencer avec une configuration puis ajuster les paramètres.
        """)
        
        st.subheader("Traitement de plusieurs images")
        st.write("""
        Lʼapplication permet de traiter plusieurs images simultanément:
        1. Téléchargez plusieurs images.
        2. Chaque image sera traitée avec les mêmes paramètres de détection.
        3. Les insectes extraits de chaque image source seront regroupés dans des sous-dossiers correspondants à l'intérieur du fichier ZIP téléchargé.
        """)
        
        st.subheader("Format des images extraites")
        st.write("""
        Les insectes extraits sont maintenant:
        1. **Rendus carrés** : Chaque image d'insecte extraite est mise au format carré par ajout de bordures blanches si nécessaire.
        2. **Sur fond blanc** : Les insectes sont isolés sur un fond blanc pur. Les versions avec fond transparent ne sont plus proposées pour simplifier.
        
        Ce format est idéal pour lʼorganisation, la présentation et l'utilisation dans des applications de machine learning.
        """)
        
        st.subheader("Utilisation de lʼauto-ajustement")
        st.write("""
        La fonctionnalité dʼauto-ajustement tente de trouver les meilleurs `Constante de seuillage adaptatif` et `Surface minimale` :
        1. Indiquez le nombre dʼinsectes attendus.
        2. Sélectionnez "Auto-ajustement".
        3. Lʼapplication teste des combinaisons de ces deux paramètres pour s'approcher du nombre souhaité. Les autres paramètres (flou, taille du bloc, morphologie) sont à régler manuellement et influencent le résultat de l'auto-ajustement.
        
        Utile si vous connaissez le nombre dʼinsectes présents et souhaitez optimiser rapidement ces deux réglages clés.
        """)
        
        st.subheader("Astuces pour de meilleurs résultats")
        st.write("""
        1. **Qualité des images**: Utilisez des images bien éclairées, avec un bon contraste entre les insectes et le fond.
        2. **Fond uniforme**: Les fonds clairs et relativement uniformes donnent les meilleurs résultats.
        3. **Filtrer par circularité**: Activez cette option pour aider à éliminer les détections non-insectes (débris, etc.) basées sur leur forme. Une valeur entre 0.3 et 0.7 est souvent un bon point de départ.
        4. **Ajustement itératif**: Affinez progressivement les paramètres en observant les résultats intermédiaires (image floutée, seuillée, morphologique) et les statistiques.
        """)
        
        st.subheader("Paramètres avancés")
        st.write("""
        - **Noyau de flou gaussien**: Lisse l'image, aide à réduire le bruit. Un noyau plus grand floute davantage.
        - **Taille du bloc adaptatif**: Taille du voisinage utilisé pour calculer le seuil adaptatif. Doit être impair.
        - **Constante de seuillage adaptatif**: Constante soustraite de la moyenne ou de la moyenne pondérée. Peut être négative.
        - **Noyau morphologique / Itérations**: Contrôlent la taille et l'agressivité des opérations morphologiques (ouverture, fermeture) qui nettoient l'image binaire.
        - **Surface minimale**: Taille minimale (en pixels) pour qu'un objet détecté soit considéré comme un insecte.
        - **Marge**: Espace ajouté autour de la boîte englobante de l'insecte lors de l'extraction.
        - **Circularité**: Filtre les objets selon leur rondeur (1.0 = cercle parfait).
        """)
        
        st.subheader("Résolution des problèmes courants")
        st.write("""
        - **Trop peu dʼinsectes détectés**: Essayez de diminuer la `Surface minimale`, d'augmenter la `Constante de seuillage adaptatif` (vers des valeurs plus positives), ou de réduire le `Noyau de flou`. Vérifiez aussi que les opérations morphologiques ne sont pas trop agressives (réduire taille/itérations du `Noyau morph.`).
        - **Trop dʼinsectes détectés (faux positifs)**: Augmentez la `Surface minimale`. Diminuez la `Constante de seuillage adaptatif` (vers des valeurs plus négatives). Activez et ajustez le filtre de `Circularité`. Augmentez la taille/itérations du `Noyau morph.` pour éliminer le bruit.
        - **Insectes fragmentés**: Diminuez la taille/itérations du `Noyau morph.` pour l'opération d'ouverture (si elle est trop agressive). Augmentez les itérations pour l'opération de fermeture ou de dilatation initiale pour mieux connecter les parties.
        - **Insectes proches fusionnés**: Augmentez la taille/itérations du `Noyau morph.` pour l'opération d'ouverture. Diminuez les opérations de fermeture ou de dilatation.
        """)

if __name__ == "__main__":
    main()

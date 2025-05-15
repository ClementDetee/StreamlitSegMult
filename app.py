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
    Returns a dictionary including 'params_used' which reflects actual parameters
    after potential internal auto-adjustment of adapt_c and min_area.
    """
    blur_kernel = params["blur_kernel"]
    adapt_block_size = params["adapt_block_size"]
    # These might be overridden by auto_adjust logic below
    adapt_c = params["adapt_c"]
    min_area = params["min_area"]
    
    morph_kernel_size = params["morph_kernel"]
    morph_iterations = params["morph_iterations"]
    auto_adjust_c_area = params["auto_adjust"] # For adapt_c and min_area
    use_circularity = params.get("use_circularity", False)
    min_circularity = params.get("min_circularity", 0.3)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if blur_kernel > 1:
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    else:
        blurred = gray
    
    # This section handles auto-adjustment of adapt_c and min_area
    if auto_adjust_c_area:
        adapt_c_values = [-5, 0, 2, 5, 8, 10, 15]
        min_area_values = [20, 30, 50, 75, 100, 150, 200, 300]
        
        best_params_auto = {"adapt_c": adapt_c, "min_area": min_area} # Start with current
        best_count_diff = float('inf')
        best_filtered_props_auto = [] # Renamed to avoid conflict
        
        for ac_auto in adapt_c_values:
            for ma_auto in min_area_values:
                thresh_auto = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, adapt_block_size, ac_auto
                )
                kernel_auto = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
                opening_auto = cv2.morphologyEx(thresh_auto, cv2.MORPH_OPEN, kernel_auto, iterations=morph_iterations)
                cleared_auto = clear_border(opening_auto)
                labels_auto = measure.label(cleared_auto)
                props_auto = measure.regionprops(labels_auto)
                
                current_filtered_props_auto = [prop for prop in props_auto if prop.area >= ma_auto]
                count_diff = abs(len(current_filtered_props_auto) - expected_insects)
                
                if count_diff < best_count_diff:
                    best_count_diff = count_diff
                    best_params_auto["adapt_c"] = ac_auto
                    best_params_auto["min_area"] = ma_auto
                    best_filtered_props_auto = current_filtered_props_auto
                    if best_count_diff == 0: # Optimization: if exact match found
                        break
            if best_count_diff == 0:
                break
        
        adapt_c = best_params_auto["adapt_c"]
        min_area = best_params_auto["min_area"]
        
        # Recalculate with best auto params for consistency
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c
        )
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        cleared = clear_border(opening)
        labels = measure.label(cleared)
        filtered_props = best_filtered_props_auto # Use the props found by the best auto params
        
    else: # Standard processing with given adapt_c and min_area
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c
        )
        connect_kernel = np.ones((5, 5), np.uint8)
        dilated_thresh = cv2.dilate(thresh, connect_kernel, iterations=2)
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        closing = cv2.morphologyEx(dilated_thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)
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

    # Store the actual parameters used for this specific processing run
    final_params_used = params.copy()
    final_params_used['adapt_c'] = adapt_c # Updated if auto_adjust_c_area was true
    final_params_used['min_area'] = min_area # Updated if auto_adjust_c_area was true
    # blur_kernel, morph_kernel_size, etc., are as per the input `params`
    
    return {
        "blurred": blurred,
        "thresh": thresh, 
        "opening": opening,
        "labels": labels,
        "filtered_props": filtered_props,
        "params_used": final_params_used 
    }

def extract_insects(image, filtered_props, margin):
    extracted_insects = []
    for i, prop in enumerate(filtered_props):
        minr, minc, maxr, maxc = prop.bbox
        minr = max(0, minr - margin)
        minc = max(0, minc - margin)
        maxr = min(image.shape[0], maxr + margin)
        maxc = min(image.shape[1], maxc + margin)

        insect_roi = image[minr:maxr, minc:maxc].copy()
        roi_height, roi_width = insect_roi.shape[:2]

        if roi_height == 0 or roi_width == 0:
            continue
            
        mask_from_coords = np.zeros((roi_height, roi_width), dtype=np.uint8)
        for coord in prop.coords:
            r_roi, c_roi = coord[0] - minr, coord[1] - minc
            if 0 <= r_roi < roi_height and 0 <= c_roi < roi_width:
                mask_from_coords[r_roi, c_roi] = 255
        
        kernel_close_initial = np.ones((5, 5), np.uint8)
        mask_step1 = cv2.morphologyEx(mask_from_coords, cv2.MORPH_CLOSE, kernel_close_initial, iterations=2)

        contours, _ = cv2.findContours(mask_step1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask_step2_filled = np.zeros_like(mask_step1)
        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 10:
                    cv2.drawContours(mask_step2_filled, [contour], -1, 255, thickness=cv2.FILLED)
        
        kernel_dilate_connect = np.ones((3, 3), np.uint8)
        mask_step3_dilated = cv2.dilate(mask_step2_filled, kernel_dilate_connect, iterations=1)
        kernel_close_holes = np.ones((7, 7), np.uint8)
        mask_step4_closed = cv2.morphologyEx(mask_step3_dilated, cv2.MORPH_CLOSE, kernel_close_holes, iterations=3)

        mask_bordered_for_floodfill = cv2.copyMakeBorder(mask_step4_closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        floodfill_aux_mask = np.zeros((roi_height + 4, roi_width + 4), dtype=np.uint8)
        cv2.floodFill(mask_bordered_for_floodfill, floodfill_aux_mask, (0, 0), 128)
        holes_mask = np.where((mask_bordered_for_floodfill != 128) & (mask_bordered_for_floodfill != 255), 255, 0).astype(np.uint8)
        holes_mask = holes_mask[1:-1, 1:-1]
        mask_step5_holes_filled = cv2.bitwise_or(mask_step4_closed, holes_mask)

        contours_final_selection, _ = cv2.findContours(mask_step5_holes_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask_step5_holes_filled)
        if contours_final_selection:
            largest_contour = max(contours_final_selection, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 20:
                 cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        kernel_smooth_final = np.ones((3, 3), np.uint8)
        final_smooth_mask = cv2.dilate(clean_mask, kernel_smooth_final, iterations=1)

        mask_3ch = cv2.cvtColor(final_smooth_mask, cv2.COLOR_GRAY2BGR)
        white_bg = np.ones_like(insect_roi, dtype=np.uint8) * 255
        insect_on_white = np.where(mask_3ch == 255, insect_roi, white_bg)
        square_insect = make_square(insect_on_white, fill_color=(255, 255, 255))
        
        extracted_insects.append({"image": square_insect, "index": i})
    return extracted_insects

def main():
    st.title("Détection et isolation dʼinsectes")
    st.write("Cette application permet de détecter des insectes sur un fond clair et de les isoler individuellement.")

    tab1, tab2 = st.tabs(["Application", "Guide dʼutilisation"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Choisissez une ou plusieurs images (glissez-déposez ou parcourez)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            all_results_data = []
            
            st.sidebar.header("Paramètres de détection")
            expected_insects = st.sidebar.number_input("Nombre dʼinsectes attendus", min_value=1, value=5, step=1)
            
            presets = {
                "Par défaut": {"blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 5, "morph_kernel": 3, "morph_iterations": 2, "min_area": 100, "margin": 15},
                "Grands insectes": {"blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 8, "morph_kernel": 5, "morph_iterations": 2, "min_area": 300, "margin": 15},
                "Petits insectes": {"blur_kernel": 3, "adapt_block_size": 15, "adapt_c": 2, "morph_kernel": 3, "morph_iterations": 1, "min_area": 30, "margin": 5},
                "Haute précision": {"blur_kernel": 5, "adapt_block_size": 25, "adapt_c": 12, "morph_kernel": 3, "morph_iterations": 2, "min_area": 150, "margin": 10},
                "Arthropodes à pattes fines": {"blur_kernel": 3, "adapt_block_size": 21, "adapt_c": 3, "morph_kernel": 3, "morph_iterations": 2, "min_area": 150, "margin": 20}
            }
            
            preset_choice = st.sidebar.selectbox("Configurations prédéfinies", ["Personnalisé", "Auto-ajustement C/Aire"] + list(presets.keys()), index=2)
            
            # Initialize params_config with base defaults, then override by preset/sliders
            params_config = presets["Par défaut"].copy() 
            auto_adjust_c_area_mode = False

            if preset_choice == "Auto-ajustement C/Aire":
                st.sidebar.info(f"Ajustement auto. de C et Aire pour {expected_insects} insectes.")
                auto_adjust_c_area_mode = True
                # User can still set base blur, block, morph for this mode
                # adapt_c and min_area from preset are starting points but will be auto-tuned by process_image
            elif preset_choice != "Personnalisé":
                params_config = presets[preset_choice].copy()
            # If "Personnalisé", params_config (initially "Par Défaut") will be modified by sliders below

            # Sliders always visible, their defaults change with preset, user can override
            params_config["blur_kernel"] = st.sidebar.slider("Noyau de flou gaussien", 1, 21, params_config["blur_kernel"], step=2, key=f"blur_slider_{preset_choice}")
            params_config["adapt_block_size"] = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, params_config["adapt_block_size"], step=2, key=f"block_slider_{preset_choice}")
            if not auto_adjust_c_area_mode: # Only show these if not in C/Area auto-adjust mode
                params_config["adapt_c"] = st.sidebar.slider("Constante de seuillage adaptatif", -10, 30, params_config["adapt_c"], key=f"c_slider_{preset_choice}")
                params_config["min_area"] = st.sidebar.slider("Surface minimale (pixels)", 10, 1000, params_config["min_area"], key=f"area_slider_{preset_choice}")
            
            params_config["morph_kernel"] = st.sidebar.slider("Taille du noyau morphologique", 1, 9, params_config["morph_kernel"], step=2, key=f"morph_k_slider_{preset_choice}")
            params_config["morph_iterations"] = st.sidebar.slider("Itérations morphologiques", 1, 5, params_config["morph_iterations"], key=f"morph_i_slider_{preset_choice}")
            params_config["margin"] = st.sidebar.slider("Marge autour des insectes", 0, 50, params_config["margin"], key=f"margin_slider_{preset_choice}")

            params_config["use_circularity"] = st.sidebar.checkbox("Filtrer par circularité", value=False)
            if params_config["use_circularity"]:
                params_config["min_circularity"] = st.sidebar.slider("Circularité minimale", 0.0, 1.0, 0.3, step=0.05)
            
            params_config["auto_adjust"] = auto_adjust_c_area_mode # Flag for process_image

            attempt_blur_auto_adjust = st.sidebar.checkbox("Activer l'auto-ajustement du Flou si compte incorrect", value=True)

            for file_index, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"### Image {file_index + 1}: {uploaded_file.name}")
                file_bytes = uploaded_file.getvalue()
                nparr = np.frombuffer(file_bytes, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption=f"Image originale - {uploaded_file.name}", use_column_width=True)

                with st.spinner(f"Traitement initial de lʼimage {file_index + 1}..."):
                    # Initial processing run with current params_config
                    processed_data = process_image(cv_image, params_config, expected_insects)
                
                num_detected_initial = len(processed_data["filtered_props"])
                
                # --- Auto-adjust Blur Kernel if count is incorrect ---
                if attempt_blur_auto_adjust and num_detected_initial != expected_insects:
                    st.info(f"Image {file_index + 1}: Nombre initial d'insectes ({num_detected_initial}) incorrect. "
                            f"Tentative d'ajustement automatique du noyau de flou...")
                    
                    blur_kernel_options = [k for k in range(1, 22, 2)] # e.g., 1, 3, ..., 21
                    
                    best_blur_kernel_found = processed_data["params_used"]["blur_kernel"]
                    min_diff_found = abs(num_detected_initial - expected_insects)
                    best_processed_data_after_blur_trials = processed_data

                    # Create a mutable copy for trial runs
                    temp_params_for_blur_trials = params_config.copy() 
                    # Ensure auto_adjust for C/Area is carried over if it was active
                    temp_params_for_blur_trials["auto_adjust"] = auto_adjust_c_area_mode 


                    for trial_blur in blur_kernel_options:
                        if trial_blur == best_blur_kernel_found and min_diff_found == abs(num_detected_initial - expected_insects): # Skip if it's the initial blur unless it's the only option
                             pass # Allow re-processing if it wasn't the initial best

                        temp_params_for_blur_trials["blur_kernel"] = trial_blur
                        with st.spinner(f"Image {file_index + 1}: Test avec noyau de flou {trial_blur}..."):
                            trial_data = process_image(cv_image, temp_params_for_blur_trials, expected_insects)
                        
                        trial_num_detected = len(trial_data["filtered_props"])
                        trial_diff = abs(trial_num_detected - expected_insects)

                        if trial_diff < min_diff_found:
                            min_diff_found = trial_diff
                            best_blur_kernel_found = trial_blur
                            best_processed_data_after_blur_trials = trial_data
                        elif trial_diff == min_diff_found: # Prefer smaller blur or original if diff is same
                            if trial_blur < best_blur_kernel_found : # Prioritize smaller blur for same diff
                                best_blur_kernel_found = trial_blur
                                best_processed_data_after_blur_trials = trial_data
                        
                        if min_diff_found == 0: # Exact match found
                            break
                    
                    # Update processed_data with the best result from blur trials
                    if best_processed_data_after_blur_trials["params_used"]["blur_kernel"] != processed_data["params_used"]["blur_kernel"] or \
                       len(best_processed_data_after_blur_trials["filtered_props"]) != num_detected_initial :
                        if min_diff_found < abs(num_detected_initial - expected_insects) or \
                           (min_diff_found == abs(num_detected_initial - expected_insects) and \
                            best_processed_data_after_blur_trials["params_used"]["blur_kernel"] != processed_data["params_used"]["blur_kernel"]):
                            st.success(f"Image {file_index + 1}: Noyau de flou ajusté à "
                                       f"{best_processed_data_after_blur_trials['params_used']['blur_kernel']}. "
                                       f"Nombre détecté : {len(best_processed_data_after_blur_trials['filtered_props'])} (Attendu: {expected_insects})")
                            processed_data = best_processed_data_after_blur_trials
                        else:
                             st.warning(f"Image {file_index + 1}: Ajustement du noyau de flou n'a pas amélioré le comptage. "
                                       f"Utilisation du résultat initial avec noyau {processed_data['params_used']['blur_kernel']}: "
                                       f"{num_detected_initial} insectes.")
                    elif min_diff_found == 0 and len(best_processed_data_after_blur_trials["filtered_props"]) == expected_insects:
                         st.success(f"Image {file_index + 1}: Comptage correct ({expected_insects}) "
                                   f"obtenu avec noyau de flou {best_processed_data_after_blur_trials['params_used']['blur_kernel']}.")
                         processed_data = best_processed_data_after_blur_trials

                # --- End of Auto-adjust Blur Kernel ---

                # Final results for this image (after any adjustments)
                final_params_used = processed_data["params_used"]
                current_filtered_props = processed_data["filtered_props"]
                num_detected_final = len(current_filtered_props)

                all_results_data.append({
                    "filename": uploaded_file.name,
                    "image": cv_image,
                    "results": processed_data # Store the finalized processed data
                })
                
                # Display info based on final_params_used
                if auto_adjust_c_area_mode:
                    st.success(f"Image {file_index + 1}: Params C/Aire auto-ajustés (Flou: {final_params_used['blur_kernel']}): "
                               f"C={final_params_used['adapt_c']}, Aire Min={final_params_used['min_area']}")
                elif final_params_used['blur_kernel'] != params_config['blur_kernel'] and attempt_blur_auto_adjust: # if blur was adjusted and wasn't from C/Area mode
                     pass # Message already shown by blur adjustment logic
                
                # Display image processing steps
                col1, col2 = st.columns(2)
                with col1:
                    st.image(processed_data["blurred"], caption=f"Floutée (Noyau: {final_params_used['blur_kernel']})", use_column_width=True)
                    st.image(processed_data["thresh"], caption=f"Seuillage (C: {final_params_used['adapt_c']})", use_column_width=True)
                with col2:
                    st.image(processed_data["opening"], caption="Après opérations morphologiques", use_column_width=True)
                    label_display = np.zeros((processed_data["labels"].shape[0], processed_data["labels"].shape[1], 3), dtype=np.uint8)
                    for prop_item in current_filtered_props:
                        color = np.random.randint(0, 255, size=3)
                        for coord in prop_item.coords:
                            label_display[coord[0], coord[1]] = color
                    st.image(label_display, caption=f"Insectes détectés: {num_detected_final} (Aire Min: {final_params_used['min_area']})", use_column_width=True)

                st.subheader(f"Statistiques de détection finales - {uploaded_file.name}")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                stat_col1.metric("Nombre dʼinsectes", num_detected_final)
                stat_col1.metric("Nombre attendu", expected_insects)
                
                if current_filtered_props:
                    areas = [prop.area for prop in current_filtered_props]
                    stat_col2.metric("Surface moyenne (px)", f"{int(np.mean(areas)) if areas else 0}")
                    stat_col3.metric("Plage de tailles (px)", f"{int(min(areas)) if areas else 0} - {int(max(areas)) if areas else 0}")
                
                diff_final = abs(num_detected_final - expected_insects)
                if diff_final == 0:
                    st.success(f"✅ Nombre exact dʼinsectes détectés: {num_detected_final}")
                elif diff_final <= 2:
                    st.warning(f"⚠️ {num_detected_final} insectes détectés (écart de {diff_final})")
                else:
                    st.error(f"❌ {num_detected_final} insectes détectés (écart important de {diff_final})")
            
                if current_filtered_props:
                    extracted_insects_list = extract_insects(cv_image, current_filtered_props, final_params_used["margin"])
                    st.write(f"Aperçu des premiers insectes isolés de {uploaded_file.name}:")
                    preview_cols = st.columns(min(5, len(extracted_insects_list)))
                    for i_prev, col_prev in enumerate(preview_cols):
                        if i_prev < len(extracted_insects_list):
                            insect_data = extracted_insects_list[i_prev]
                            col_prev.image(cv2.cvtColor(insect_data["image"], cv2.COLOR_BGR2RGB), caption=f"Insecte {insect_data['index']+1}", use_column_width=True)
                st.markdown("---")
            
            if st.button("Extraire et télécharger tous les insectes isolés (carrés, fond blanc)"):
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "insectes_isoles.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for result_item in all_results_data:
                        image_orig = result_item["image"]
                        filename_base = os.path.splitext(result_item["filename"])[0]
                        # Use the finalized props and margin for extraction
                        props_for_extraction = result_item["results"]["filtered_props"]
                        margin_for_extraction = result_item["results"]["params_used"]["margin"]
                        
                        extracted_insects_for_zip = extract_insects(image_orig, props_for_extraction, margin_for_extraction)
                        for insect_detail in extracted_insects_for_zip:
                            insect_img_square = insect_detail["image"]
                            insect_idx = insect_detail["index"]
                            temp_img_filename = f"{filename_base}_insect_{insect_idx+1}.jpg"
                            temp_img_path = os.path.join(temp_dir, temp_img_filename)
                            cv2.imwrite(temp_img_path, insect_img_square)
                            zipf.write(temp_img_path, os.path.join(filename_base, temp_img_filename))
                with open(zip_path, "rb") as f:
                    bytes_data = f.read()
                    b64 = base64.b64encode(bytes_data).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="insectes_isoles.zip">Télécharger tous les insectes isolés (ZIP)</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    with tab2:
        st.header("Guide dʼoptimisation des paramètres")
        st.subheader("Configurations prédéfinies")
        st.write("""
        - **Par défaut**: Configuration équilibrée.
        - **Grands insectes / Petits insectes / Haute précision / Arthropodes à pattes fines**: Optimisations spécifiques.
        - **Auto-ajustement C/Aire**: Ajuste `Constante de seuillage adaptatif` et `Surface minimale` pour atteindre le nombre d'insectes attendu. Les autres paramètres (flou, bloc, morphologie) sont réglés manuellement via les curseurs et servent de base à cet ajustement.
        """)
        
        st.subheader("Auto-ajustement du Noyau de Flou")
        st.write("""
        Si la case "Activer l'auto-ajustement du Flou si compte incorrect" est cochée dans la barre latérale :
        1. Après un premier essai de détection sur une image, si le nombre d'insectes trouvés ne correspond pas au nombre attendu, l'application va automatiquement tester différentes valeurs pour le "Noyau de flou gaussien".
        2. Pour chaque valeur de flou testée, si le mode "Auto-ajustement C/Aire" est actif, les paramètres `C` et `Aire` seront également ré-optimisés.
        3. L'application choisira la valeur de flou (et les `C`/`Aire` associés si en mode auto) qui donne le nombre d'insectes le plus proche du nombre attendu.
        4. Ce processus est effectué pour chaque image individuellement, permettant d'adapter le flou aux spécificités de chaque photo.
        Un message indiquera si le flou a été ajusté pour une image et quelle valeur a été retenue.
        """)

        st.subheader("Traitement de plusieurs images")
        st.write("""
        L'application permet de traiter plusieurs images en une seule fois. Chaque image est traitée séquentiellement.
        Les paramètres de détection globaux sont appliqués à chaque image, mais l'auto-ajustement du flou (si activé) peut modifier le noyau de flou spécifiquement pour chaque image afin d'optimiser le comptage.
        Les insectes extraits sont regroupés par image source dans le fichier ZIP.
        """)
        
        st.subheader("Format des images extraites")
        st.write("""
        Les images d'insectes extraites sont carrées (ratio 1:1 avec ajout de bordures blanches) et sur fond blanc.
        """)
        
        st.subheader("Astuces pour de meilleurs résultats")
        st.write("""
        - **Qualité des images**: Bon éclairage, contraste suffisant.
        - **Fond uniforme**: Les fonds clairs et uniformes sont préférables.
        - **Filtrer par circularité**: Utile pour éliminer les débris.
        - **Ajustement itératif**: Observez les résultats intermédiaires et les messages d'auto-ajustement.
        """)
        
        st.subheader("Paramètres avancés")
        st.write("""
        - **Noyau de flou gaussien**: Lisse l'image. Peut être auto-ajusté.
        - **Taille du bloc adaptatif**: Voisinage pour le seuil adaptatif.
        - **Constante de seuillage adaptatif (C)**: Ajuste le seuil. Peut être auto-ajusté avec "Auto-ajustement C/Aire".
        - **Surface minimale (Aire)**: Filtre les petits objets. Peut être auto-ajusté avec "Auto-ajustement C/Aire".
        - **Noyau morphologique / Itérations**: Nettoient l'image binaire.
        - **Marge**: Espace ajouté autour de l'insecte extrait.
        - **Circularité**: Filtre par forme.
        """)
        
        st.subheader("Résolution des problèmes courants")
        st.write("""
        - **Comptage incorrect persistant**: Si l'auto-ajustement du flou et/ou C/Aire ne suffit pas, essayez d'ajuster manuellement la `Taille du bloc adaptatif` ou les paramètres morphologiques. La qualité de l'image initiale est cruciale.
        - **Trop peu dʼinsectes**: Si les auto-ajustements sont désactivés, essayez de diminuer `Surface minimale`, d'augmenter `Constante C`, ou de réduire le `Noyau de flou`.
        - **Trop dʼinsectes (faux positifs)**: Si les auto-ajustements sont désactivés, augmentez `Surface minimale`, diminuez `Constante C`, ou activez `Circularité`.
        - **Insectes fragmentés/fusionnés**: Ajustez les paramètres morphologiques (`Noyau morph.`, `Itérations morph.`).
        """)

if __name__ == "__main__":
    main()

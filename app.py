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
    height, width = image.shape[:2]
    max_side = max(height, width)
    top = (max_side - height) // 2
    bottom = max_side - height - top
    left = (max_side - width) // 2
    right = max_side - width - left
    square_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    return square_image

def process_image(image, params, expected_insects_for_image_info=0): # expected_insects_for_image_info is mostly for context now
    """
    Traite une image selon les paramètres fournis et retourne les résultats.
    Includes relative size filtering.
    """
    blur_kernel = params["blur_kernel"]
    adapt_block_size = params["adapt_block_size"]
    adapt_c = params["adapt_c"]
    min_area = params["min_area"]
    
    morph_kernel_size = params["morph_kernel"]
    morph_iterations = params["morph_iterations"]
    auto_adjust_c_area_internally = params.get("auto_adjust_for_internal_tune", False) # Specific flag for internal C/Area tuning
    use_circularity = params.get("use_circularity", False)
    min_circularity = params.get("min_circularity", 0.3)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if blur_kernel > 1:
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    else:
        blurred = gray
    
    current_adapt_c = adapt_c
    current_min_area = min_area

    if auto_adjust_c_area_internally: # This block is for when process_image is called during global param tuning
        adapt_c_values = [-5, 0, 2, 5, 8, 10, 15]
        min_area_values = [20, 30, 50, 75, 100, 150, 200, 300]
        best_params_auto = {"adapt_c": current_adapt_c, "min_area": current_min_area}
        best_count_diff = float('inf')
        best_filtered_props_auto = []
        
        for ac_auto in adapt_c_values:
            for ma_auto in min_area_values:
                thresh_auto = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block_size, ac_auto)
                kernel_auto = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
                opening_auto = cv2.morphologyEx(thresh_auto, cv2.MORPH_OPEN, kernel_auto, iterations=morph_iterations)
                cleared_auto = clear_border(opening_auto)
                labels_auto = measure.label(cleared_auto)
                props_auto = measure.regionprops(labels_auto)
                current_filtered_props_auto = [prop for prop in props_auto if prop.area >= ma_auto]
                count_diff = abs(len(current_filtered_props_auto) - expected_insects_for_image_info) # Use per-image target here
                
                if count_diff < best_count_diff:
                    best_count_diff = count_diff
                    best_params_auto["adapt_c"] = ac_auto
                    best_params_auto["min_area"] = ma_auto
                    best_filtered_props_auto = current_filtered_props_auto
                    if best_count_diff == 0: break
            if best_count_diff == 0: break
        
        current_adapt_c = best_params_auto["adapt_c"]
        current_min_area = best_params_auto["min_area"]
        # Use props from this auto-tuning for the next steps
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block_size, current_adapt_c)
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        cleared = clear_border(opening)
        labels = measure.label(cleared)
        filtered_props = best_filtered_props_auto
    else: # Standard processing with fixed C and Area
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block_size, current_adapt_c)
        connect_kernel = np.ones((5, 5), np.uint8)
        dilated_thresh = cv2.dilate(thresh, connect_kernel, iterations=2)
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        closing = cv2.morphologyEx(dilated_thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)
        cleared = clear_border(opening)
        labels = measure.label(cleared)
        props = measure.regionprops(labels)
        pre_filter_props = [prop for prop in props if prop.area >= current_min_area] # Initial filter by min_area

        if use_circularity:
            filtered_props = []
            for prop in pre_filter_props: # Apply circularity on props already filtered by min_area
                perimeter = prop.perimeter
                if perimeter > 0:
                    circularity = 4 * np.pi * prop.area / (perimeter * perimeter)
                    if circularity >= min_circularity:
                        filtered_props.append(prop)
        else:
            filtered_props = pre_filter_props


    # --- Relative Size Filter (Dirt Rejection) ---
    if len(filtered_props) > 1: # Only apply if multiple objects to compare
        areas = [prop.area for prop in filtered_props]
        if areas: # Ensure areas list is not empty
            avg_area = np.mean(areas)
            # Only apply relative filter if average area is somewhat substantial,
            # and not just an average of tiny specks.
            if avg_area > max(2 * current_min_area, 50): # Heuristic: avg area should be > 2x min_area or 50 abs
                relative_threshold_area = 0.1 * avg_area
                
                final_filtered_props_after_relative = []
                for prop in filtered_props:
                    # Object must be >= relative threshold AND still >= original min_area (redundant but safe)
                    if prop.area >= relative_threshold_area and prop.area >= current_min_area :
                        final_filtered_props_after_relative.append(prop)
                filtered_props = final_filtered_props_after_relative
    # --- End of Relative Size Filter ---

    final_params_used = params.copy()
    final_params_used['adapt_c'] = current_adapt_c
    final_params_used['min_area'] = current_min_area 
    final_params_used['blur_kernel'] = blur_kernel # Reflects the blur used for this run
    
    return {
        "blurred": blurred, "thresh": thresh, "opening": opening, "labels": labels,
        "filtered_props": filtered_props, "params_used": final_params_used 
    }


def extract_insects(image, filtered_props, margin):
    # (This function remains largely the same as before)
    extracted_insects = []
    for i, prop in enumerate(filtered_props):
        minr, minc, maxr, maxc = prop.bbox
        minr = max(0, minr - margin)
        minc = max(0, minc - margin)
        maxr = min(image.shape[0], maxr + margin)
        maxc = min(image.shape[1], maxc + margin)

        insect_roi = image[minr:maxr, minc:maxc].copy()
        roi_height, roi_width = insect_roi.shape[:2]

        if roi_height == 0 or roi_width == 0: continue
            
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
    st.write("Application pour la détection globale d'insectes sur plusieurs images.")

    tab1, tab2 = st.tabs(["Application", "Guide dʼutilisation"])
    
    with tab1:
        uploaded_files = st.file_uploader("Choisissez une ou plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            all_images_results = [] # Renamed
            grand_total_detected_insects = 0
            
            st.sidebar.header("Paramètres de détection Globaux")
            expected_insects_grand_total = st.sidebar.number_input("Nombre total dʼinsectes attendus (toutes images)", min_value=1, value=len(uploaded_files) * 3, step=1) # Default guess
            
            presets = {
                "Par défaut": {"blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 5, "morph_kernel": 3, "morph_iterations": 2, "min_area": 100, "margin": 15},
                "Grands insectes": {"blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 8, "morph_kernel": 5, "morph_iterations": 2, "min_area": 300, "margin": 15},
                # ... other presets
            }
            preset_choice = st.sidebar.selectbox("Configurations prédéfinies", ["Personnalisé", "Auto-ajustement C/Aire Global"] + list(presets.keys()), index=2)
            
            # Initialize params_config with base defaults, then override by preset/sliders
            params_config_globally = presets["Par défaut"].copy() 
            # Flag to indicate if C/Area auto-adjustment should run on the first image
            should_auto_adjust_c_area_globally = False 

            if preset_choice == "Auto-ajustement C/Aire Global":
                should_auto_adjust_c_area_globally = True
            elif preset_choice != "Personnalisé":
                params_config_globally = presets[preset_choice].copy()

            # Sliders always visible, their defaults change with preset, user can override
            # These will be the initial values for global parameter determination if auto-adjust is on
            params_config_globally["blur_kernel"] = st.sidebar.slider("Noyau de flou gaussien", 1, 21, params_config_globally["blur_kernel"], step=2, key=f"blur_glob_{preset_choice}")
            params_config_globally["adapt_block_size"] = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, params_config_globally["adapt_block_size"], step=2, key=f"block_glob_{preset_choice}")
            
            # adapt_c and min_area sliders are shown, but their values might be overridden if global C/Area tuning is on
            params_config_globally["adapt_c"] = st.sidebar.slider("Constante de seuillage C", -10, 30, params_config_globally["adapt_c"], key=f"c_glob_{preset_choice}")
            params_config_globally["min_area"] = st.sidebar.slider("Surface minimale Aire", 10, 1000, params_config_globally["min_area"], key=f"area_glob_{preset_choice}")
            
            params_config_globally["morph_kernel"] = st.sidebar.slider("Noyau morphologique", 1, 9, params_config_globally["morph_kernel"], step=2, key=f"morph_k_glob_{preset_choice}")
            params_config_globally["morph_iterations"] = st.sidebar.slider("Itérations morphologiques", 1, 5, params_config_globally["morph_iterations"], key=f"morph_i_glob_{preset_choice}")
            params_config_globally["margin"] = st.sidebar.slider("Marge autour des insectes", 0, 50, params_config_globally["margin"], key=f"margin_glob_{preset_choice}")

            params_config_globally["use_circularity"] = st.sidebar.checkbox("Filtrer par circularité", value=False)
            if params_config_globally["use_circularity"]:
                params_config_globally["min_circularity"] = st.sidebar.slider("Circularité minimale", 0.0, 1.0, 0.3, step=0.05)
            
            should_auto_adjust_blur_globally = st.sidebar.checkbox("Auto-ajustement du Flou Global (sur 1ère image)", value=False)

            # --- Determine Global Parameters based on First Image ---
            st.markdown("---")
            st.subheader("Détermination des Paramètres Globaux (basée sur la 1ère image)")
            
            # Make a working copy for global tuning
            tuned_params_for_all_images = params_config_globally.copy() 

            first_image_obj = uploaded_files[0]
            first_image_bytes = first_image_obj.getvalue()
            first_nparr = np.frombuffer(first_image_bytes, np.uint8)
            first_cv_image = cv2.imdecode(first_nparr, cv2.IMREAD_COLOR)
            
            # Target for tuning on the first image
            expected_for_first_image_tuning = max(1, round(expected_insects_grand_total / len(uploaded_files)))
            st.write(f"Image de référence pour l'ajustement global: {first_image_obj.name} (cible pour cette image: ~{expected_for_first_image_tuning} insectes)")

            # 1. Global C/Area Tuning (if selected)
            if should_auto_adjust_c_area_globally:
                with st.spinner("Ajustement global C/Aire en cours..."):
                    params_for_c_area_tune = tuned_params_for_all_images.copy()
                    params_for_c_area_tune["auto_adjust_for_internal_tune"] = True # Enable internal C/Area tuning
                    
                    c_area_tune_data = process_image(first_cv_image, params_for_c_area_tune, expected_for_first_image_tuning)
                    tuned_params_for_all_images["adapt_c"] = c_area_tune_data["params_used"]["adapt_c"]
                    tuned_params_for_all_images["min_area"] = c_area_tune_data["params_used"]["min_area"]
                st.success(f"Global C/Aire déterminés: C={tuned_params_for_all_images['adapt_c']}, Aire Min={tuned_params_for_all_images['min_area']}")

            # 2. Global Blur Tuning (if selected) - uses potentially updated C/Area from above
            if should_auto_adjust_blur_globally:
                with st.spinner("Ajustement global du Flou en cours..."):
                    blur_kernel_options = [k for k in range(1, 22, 2)]
                    best_blur_found_globally = tuned_params_for_all_images["blur_kernel"]
                    min_diff_for_global_blur = float('inf')
                    
                    # Store C/Area that might get co-tuned with blur
                    c_after_blur_tune = tuned_params_for_all_images["adapt_c"]
                    area_after_blur_tune = tuned_params_for_all_images["min_area"]

                    for trial_blur in blur_kernel_options:
                        params_for_blur_trial = tuned_params_for_all_images.copy()
                        params_for_blur_trial["blur_kernel"] = trial_blur
                        # If C/Area was globally tuned, it's now fixed. If not, and user selected "Auto C/Aire Global",
                        # then internal C/Area tuning should happen *with* this trial blur.
                        params_for_blur_trial["auto_adjust_for_internal_tune"] = should_auto_adjust_c_area_globally
                        
                        trial_data = process_image(first_cv_image, params_for_blur_trial, expected_for_first_image_tuning)
                        trial_num_detected = len(trial_data["filtered_props"])
                        trial_diff = abs(trial_num_detected - expected_for_first_image_tuning)

                        if trial_diff < min_diff_for_global_blur:
                            min_diff_for_global_blur = trial_diff
                            best_blur_found_globally = trial_blur
                            if should_auto_adjust_c_area_globally: # If C/Area were co-tuned
                                c_after_blur_tune = trial_data["params_used"]["adapt_c"]
                                area_after_blur_tune = trial_data["params_used"]["min_area"]
                        elif trial_diff == min_diff_for_global_blur and trial_blur < best_blur_found_globally:
                            best_blur_found_globally = trial_blur
                            if should_auto_adjust_c_area_globally:
                                c_after_blur_tune = trial_data["params_used"]["adapt_c"]
                                area_after_blur_tune = trial_data["params_used"]["min_area"]
                        if min_diff_for_global_blur == 0: break
                    
                    tuned_params_for_all_images["blur_kernel"] = best_blur_found_globally
                    if should_auto_adjust_c_area_globally:
                        tuned_params_for_all_images["adapt_c"] = c_after_blur_tune
                        tuned_params_for_all_images["min_area"] = area_after_blur_tune
                        st.success(f"Global Flou déterminé: {tuned_params_for_all_images['blur_kernel']}. "
                                   f"Global C/Aire (après Flou): C={tuned_params_for_all_images['adapt_c']}, Aire Min={tuned_params_for_all_images['min_area']}")
                    else:
                        st.success(f"Global Flou déterminé: {tuned_params_for_all_images['blur_kernel']}.")
            
            # Ensure internal tuning flag is OFF for actual batch processing
            tuned_params_for_all_images["auto_adjust_for_internal_tune"] = False
            st.markdown("---")
            st.subheader("Traitement des Images avec Paramètres Globaux")
            st.json(tuned_params_for_all_images) # Show the final global params

            # --- Process all images with the determined global parameters ---
            for file_index, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"#### Image {file_index + 1}: {uploaded_file.name}")
                file_bytes = uploaded_file.getvalue()
                nparr = np.frombuffer(file_bytes, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                with st.spinner(f"Traitement image {file_index + 1} avec params globaux..."):
                    # Process with globally determined parameters, internal C/Area tuning is OFF now
                    processed_data = process_image(cv_image, tuned_params_for_all_images, 0) # Expected count not used for tuning here
                
                current_filtered_props = processed_data["filtered_props"]
                num_detected_this_image = len(current_filtered_props)
                grand_total_detected_insects += num_detected_this_image

                all_images_results.append({
                    "filename": uploaded_file.name, "image": cv_image, "results": processed_data
                })
                
                # Simplified Display
                disp_col1, disp_col2 = st.columns(2)
                with disp_col1:
                    st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption=f"Originale", width=300) # Smaller original
                
                with disp_col2:
                    label_display = np.zeros((processed_data["labels"].shape[0], processed_data["labels"].shape[1], 3), dtype=np.uint8)
                    for prop_item in current_filtered_props:
                        color = np.random.randint(0, 255, size=3)
                        for coord in prop_item.coords:
                            label_display[coord[0], coord[1]] = color
                    st.image(label_display, caption=f"Insectes détectés: {num_detected_this_image} "
                                                   f"(Aire Min Globale: {tuned_params_for_all_images['min_area']})", 
                             use_column_width=True)

                # Per-image stats
                stat_col1, stat_col2 = st.columns(2)
                stat_col1.metric(f"Insectes détectés (Image {file_index+1})", num_detected_this_image)
                if current_filtered_props:
                    areas_this_image = [prop.area for prop in current_filtered_props]
                    stat_col2.metric(f"Surface moyenne (Image {file_index+1}, px)", f"{int(np.mean(areas_this_image)) if areas_this_image else 0}")
                else:
                    stat_col2.metric(f"Surface moyenne (Image {file_index+1}, px)", "N/A")
                
                # Preview of extracted insects (optional, can be verbose)
                if current_filtered_props and num_detected_this_image > 0 :
                    extracted_preview = extract_insects(cv_image, current_filtered_props[:min(3, num_detected_this_image)], tuned_params_for_all_images["margin"])
                    if extracted_preview:
                        st.write("Aperçu des 1ers insectes extraits:")
                        preview_cols_ext = st.columns(len(extracted_preview))
                        for i_ext, insect_ext_data in enumerate(extracted_preview):
                            preview_cols_ext[i_ext].image(cv2.cvtColor(insect_ext_data["image"], cv2.COLOR_BGR2RGB), width=100)
                st.markdown("---")

            # --- Grand Total Results ---
            st.header("Résultats Globaux Finaux")
            st.metric("Nombre TOTAL d'insectes attendus (toutes images)", expected_insects_grand_total)
            st.metric("Nombre TOTAL d'insectes détectés (toutes images)", grand_total_detected_insects)
            
            diff_grand_total = abs(grand_total_detected_insects - expected_insects_grand_total)
            if diff_grand_total == 0:
                st.success("✅ Succès! Nombre total d'insectes détectés correspond au nombre attendu.")
            elif diff_grand_total <= 0.1 * expected_insects_grand_total : # Allow 10% tolerance
                st.warning(f"⚠️ Attention: {grand_total_detected_insects} insectes détectés au total (attendu: {expected_insects_grand_total}, écart de {diff_grand_total}).")
            else:
                st.error(f"❌ Erreur: {grand_total_detected_insects} insectes détectés au total (attendu: {expected_insects_grand_total}, écart important de {diff_grand_total}).")

            if st.button("Extraire et télécharger TOUS les insectes isolés (carrés, fond blanc)"):
                # (Download logic remains the same, ensure it uses `tuned_params_for_all_images["margin"]`)
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "insectes_isoles_batch.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for result_item in all_images_results:
                        image_orig = result_item["image"]
                        filename_base = os.path.splitext(result_item["filename"])[0]
                        props_for_extraction = result_item["results"]["filtered_props"]
                        # Use the globally determined margin
                        margin_for_extraction = tuned_params_for_all_images["margin"] 
                        
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
                    href = f'<a href="data:application/zip;base64,{b64}" download="insectes_isoles_batch.zip">Télécharger tous les insectes isolés (ZIP)</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    with tab2: # Update Guide
        st.header("Guide dʼoptimisation des paramètres")
        st.subheader("Approche Globale de Traitement")
        st.write("""
        Cette version de l'application adopte une approche de **paramètres globaux**. 
        - Les paramètres de détection (Flou, Constante C, Aire Minimale, etc.) sont déterminés **une seule fois** au début du traitement.
        - Si les options "Auto-ajustement C/Aire Global" ou "Auto-ajustement du Flou Global" sont activées, ces paramètres sont optimisés en se basant sur la **première image** de votre lot. La cible pour cette optimisation est le `Nombre total dʼinsectes attendus / nombre d'images`.
        - Une fois déterminés, ces paramètres sont appliqués **identiquement à toutes les images** du lot.
        - Le `Nombre total dʼinsectes attendus` que vous spécifiez est pour l'ensemble des images.
        """)

        st.subheader("Filtrage Relatif des Petits Objets (Anti-Saleté)")
        st.write("""
        Pour réduire la détection de petites saletés :
        1. Après une détection initiale sur une image, si plusieurs objets sont trouvés, leur surface moyenne est calculée.
        2. Tout objet dont la surface est inférieure à 10% de cette moyenne est automatiquement écarté.
        3. Ce filtre est appliqué **par image**, car la "taille moyenne" est relative aux insectes de cette image spécifique. Il s'ajoute au filtre global `Surface minimale Aire`.
        """)

        st.subheader("Affichage Simplifié")
        st.write("""
        Pour chaque image traitée, l'affichage principal montre :
        - L'image originale (redimensionnée pour être plus petite).
        - L'image finale avec les "Insectes détectés" et leur nombre pour cette image.
        Les étapes intermédiaires de traitement (flou, seuillage, etc.) ne sont plus affichées par défaut pour alléger l'interface.
        """)

        st.subheader("Configurations Prédéfinies et Auto-Ajustement Global")
        st.write("""
        - **Auto-ajustement C/Aire Global**: Si activé, `Constante C` et `Surface minimale Aire` sont optimisées sur la 1ère image.
        - **Auto-ajustement du Flou Global**: Si activé, `Noyau de flou` est optimisé sur la 1ère image (peut aussi ré-optimiser C/Aire si l'option C/Aire est aussi active).
        - Les autres paramètres sont pris des curseurs.
        """)

        st.subheader("Astuces pour de Meilleurs Résultats avec l'Approche Globale")
        st.write("""
        - **Qualité de la 1ère image**: Elle est cruciale si vous utilisez les auto-ajustements globaux. Assurez-vous qu'elle soit représentative de votre lot.
        - **Homogénéité du lot**: Si vos images sont très différentes (ex: types d'insectes, éclairage, zoom très variés), un jeu unique de paramètres globaux pourrait ne pas être optimal pour toutes. Dans ce cas, traitez des sous-lots plus homogènes séparément.
        - **Nombre total attendu**: Fournissez une estimation raisonnable pour le `Nombre total dʼinsectes attendus` sur l'ensemble du lot.
        """)
        # ... (other relevant sections of the guide can be kept or adapted)

if __name__ == "__main__":
    main()

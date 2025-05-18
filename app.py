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
import tensorflow as tf # Ajout pour Keras

# --- Configuration du Modèle d'Identification ---
MODEL_PATH = "keras_model.h5"  # Assurez-vous que ce fichier est dans votre repo
LABELS_PATH = "labels.txt"    # Assurez-vous que ce fichier est dans votre repo
MODEL_INPUT_SIZE = (224, 224) # Taille d'entrée attendue par votre modèle Teachable Machine (souvent 224x224)

# --- Fonctions existantes (make_square, process_image, extract_insects) ---
# (Votre code existant pour make_square, process_image, extract_insects reste ici)
# ... (j'omets ces fonctions pour la brièveté, elles ne changent pas structurellement)
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

        extracted_insects.append({"image": square_insect, "index": i, "original_prop": prop}) # Ajout original_prop pour référence
    return extracted_insects


# --- Nouvelles fonctions pour le modèle Keras ---
@st.cache_resource # Cache la ressource pour ne la charger qu'une fois
def load_keras_model_and_labels(model_path, labels_path):
    """Charge le modèle Keras et les noms des labels."""
    try:
        model = tf.keras.models.load_model(model_path, compile=False) # compile=False est souvent nécessaire pour les modèles TM
        with open(labels_path, "r") as f:
            class_names = [line.strip().split(" ", 1)[1] if " " in line.strip() else line.strip() for line in f.readlines()] # Gère "0 Abeille" ou "Abeille"
        return model, class_names
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle ou des labels: {e}")
        st.error(f"Vérifiez que les fichiers '{model_path}' et '{labels_path}' existent et sont corrects.")
        return None, None

def predict_insect(image_cv2, model, class_names, input_size):
    """
    Prétraite une image et prédit la classe d'insecte.
    image_cv2: image OpenCV (BGR) de l'insecte isolé.
    """
    if model is None or class_names is None:
        return "Erreur Modèle", 0.0, []

    # 1. Redimensionner à la taille d'entrée du modèle
    # Votre fonction make_square a déjà rendu l'image carrée.
    # Maintenant, redimensionnez-la à la taille exacte attendue par le modèle.
    img_resized = cv2.resize(image_cv2, input_size, interpolation=cv2.INTER_AREA)

    # 2. Convertir en tableau NumPy et normaliser
    # Teachable Machine normalise typiquement les images entre -1 et 1.
    # (image / 127.5) - 1
    image_array = np.asarray(img_resized, dtype=np.float32)
    image_array = (image_array / 127.5) - 1.0

    # 3. Créer le batch (une seule image)
    data = np.ndarray(shape=(1, input_size[0], input_size[1], 3), dtype=np.float32)
    data[0] = image_array

    # 4. Faire la prédiction
    prediction = model.predict(data)
    
    # 5. Interpréter les résultats
    # prediction[0] contient les probabilités pour chaque classe
    predicted_class_index = np.argmax(prediction[0])
    confidence_score = prediction[0][predicted_class_index]
    label_name = class_names[predicted_class_index]
    
    return label_name, confidence_score, prediction[0]


def main():
    st.title("Détection, isolation et identification dʼinsectes") # Titre mis à jour
    st.write("Application pour la détection globale et l'identification d'insectes sur plusieurs images.")

    # --- Chargement du modèle Keras et des labels ---
    model, class_names = load_keras_model_and_labels(MODEL_PATH, LABELS_PATH)
    if model is None: # Si le chargement échoue, on s'arrête là pour l'identification
        st.warning("Le modèle d'identification n'a pas pu être chargé. La fonctionnalité d'identification sera désactivée.")

    tab1, tab2, tab3 = st.tabs(["Segmentation", "Identification", "Guide dʼutilisation"]) # Ajout onglet Identification
    
    with tab1: # Onglet Segmentation
        st.header("Phase 1 : Détection et Segmentation des insectes")
        uploaded_files = st.file_uploader("Choisissez une ou plusieurs images pour la segmentation", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            # Initialiser/Réinitialiser 'all_images_results' dans la session si les fichiers changent
            if "uploaded_files_names" not in st.session_state or \
               st.session_state.uploaded_files_names != [f.name for f in uploaded_files]:
                st.session_state.all_images_results = []
                st.session_state.uploaded_files_names = [f.name for f in uploaded_files]
                st.session_state.segmentation_done = False # Indicateur que la segmentation n'est pas (encore) faite

            if not st.session_state.get('segmentation_done', False): # Exécute la segmentation si pas encore faite pour ce lot
                all_images_results_temp = []
                grand_total_detected_insects = 0
                
                # ... (tout votre code de configuration des paramètres de segmentation ici)
                st.sidebar.header("Paramètres de détection Globaux")
                expected_insects_grand_total = st.sidebar.number_input("Nombre total dʼinsectes attendus (toutes images)", min_value=1, value=len(uploaded_files) * 3, step=1) # Default guess

                presets = {
                    "Par défaut": {"blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 5, "morph_kernel": 3, "morph_iterations": 2, "min_area": 100, "margin": 15},
                    "Grands insectes": {"blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 8, "morph_kernel": 5, "morph_iterations": 2, "min_area": 300, "margin": 15},
                }
                preset_choice = st.sidebar.selectbox("Configurations prédéfinies", ["Personnalisé", "Auto-ajustement C/Aire Global"] + list(presets.keys()), index=2)

                params_config_globally = presets["Par défaut"].copy()
                should_auto_adjust_c_area_globally = False

                if preset_choice == "Auto-ajustement C/Aire Global":
                    should_auto_adjust_c_area_globally = True
                elif preset_choice != "Personnalisé":
                    params_config_globally = presets[preset_choice].copy()

                params_config_globally["blur_kernel"] = st.sidebar.slider("Noyau de flou gaussien", 1, 21, params_config_globally["blur_kernel"], step=2, key=f"blur_glob_{preset_choice}")
                params_config_globally["adapt_block_size"] = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, params_config_globally["adapt_block_size"], step=2, key=f"block_glob_{preset_choice}")
                params_config_globally["adapt_c"] = st.sidebar.slider("Constante de seuillage C", -10, 30, params_config_globally["adapt_c"], key=f"c_glob_{preset_choice}")
                params_config_globally["min_area"] = st.sidebar.slider("Surface minimale Aire", 10, 1000, params_config_globally["min_area"], key=f"area_glob_{preset_choice}")
                params_config_globally["morph_kernel"] = st.sidebar.slider("Noyau morphologique", 1, 9, params_config_globally["morph_kernel"], step=2, key=f"morph_k_glob_{preset_choice}")
                params_config_globally["morph_iterations"] = st.sidebar.slider("Itérations morphologiques", 1, 5, params_config_globally["morph_iterations"], key=f"morph_i_glob_{preset_choice}")
                params_config_globally["margin"] = st.sidebar.slider("Marge autour des insectes", 0, 50, params_config_globally["margin"], key=f"margin_glob_{preset_choice}")
                params_config_globally["use_circularity"] = st.sidebar.checkbox("Filtrer par circularité", value=False)
                if params_config_globally["use_circularity"]:
                    params_config_globally["min_circularity"] = st.sidebar.slider("Circularité minimale", 0.0, 1.0, 0.3, step=0.05)
                should_auto_adjust_blur_globally = st.sidebar.checkbox("Auto-ajustement du Flou Global (sur 1ère image)", value=False)

                st.markdown("---")
                st.subheader("Détermination des Paramètres Globaux (basée sur la 1ère image)")

                tuned_params_for_all_images = params_config_globally.copy()
                first_image_obj = uploaded_files[0]
                first_image_bytes = first_image_obj.getvalue()
                first_nparr = np.frombuffer(first_image_bytes, np.uint8)
                first_cv_image = cv2.imdecode(first_nparr, cv2.IMREAD_COLOR)
                expected_for_first_image_tuning = max(1, round(expected_insects_grand_total / len(uploaded_files)))
                st.write(f"Image de référence pour l'ajustement global: {first_image_obj.name} (cible pour cette image: ~{expected_for_first_image_tuning} insectes)")

                if should_auto_adjust_c_area_globally:
                    with st.spinner("Ajustement global C/Aire en cours..."):
                        params_for_c_area_tune = tuned_params_for_all_images.copy()
                        params_for_c_area_tune["auto_adjust_for_internal_tune"] = True
                        c_area_tune_data = process_image(first_cv_image, params_for_c_area_tune, expected_for_first_image_tuning)
                        tuned_params_for_all_images["adapt_c"] = c_area_tune_data["params_used"]["adapt_c"]
                        tuned_params_for_all_images["min_area"] = c_area_tune_data["params_used"]["min_area"]
                    st.success(f"Global C/Aire déterminés: C={tuned_params_for_all_images['adapt_c']}, Aire Min={tuned_params_for_all_images['min_area']}")

                if should_auto_adjust_blur_globally:
                    with st.spinner("Ajustement global du Flou en cours..."):
                        blur_kernel_options = [k for k in range(1, 22, 2)]
                        best_blur_found_globally = tuned_params_for_all_images["blur_kernel"]
                        min_diff_for_global_blur = float('inf')
                        c_after_blur_tune = tuned_params_for_all_images["adapt_c"]
                        area_after_blur_tune = tuned_params_for_all_images["min_area"]
                        for trial_blur in blur_kernel_options:
                            params_for_blur_trial = tuned_params_for_all_images.copy()
                            params_for_blur_trial["blur_kernel"] = trial_blur
                            params_for_blur_trial["auto_adjust_for_internal_tune"] = should_auto_adjust_c_area_globally
                            trial_data = process_image(first_cv_image, params_for_blur_trial, expected_for_first_image_tuning)
                            trial_num_detected = len(trial_data["filtered_props"])
                            trial_diff = abs(trial_num_detected - expected_for_first_image_tuning)
                            if trial_diff < min_diff_for_global_blur:
                                min_diff_for_global_blur = trial_diff
                                best_blur_found_globally = trial_blur
                                if should_auto_adjust_c_area_globally:
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
                            st.success(f"Global Flou déterminé: {tuned_params_for_all_images['blur_kernel']}. Global C/Aire (après Flou): C={tuned_params_for_all_images['adapt_c']}, Aire Min={tuned_params_for_all_images['min_area']}")
                        else:
                            st.success(f"Global Flou déterminé: {tuned_params_for_all_images['blur_kernel']}.")
                
                tuned_params_for_all_images["auto_adjust_for_internal_tune"] = False
                st.markdown("---")
                st.subheader("Traitement des Images avec Paramètres Globaux")
                st.json(tuned_params_for_all_images) # Show the final global params

                # --- Process all images with the determined global parameters ---
                for file_index, uploaded_file in enumerate(uploaded_files):
                    st.markdown(f"#### Image {file_index + 1}: {uploaded_file.name}")
                    file_bytes = uploaded_file.getvalue() # BytesIO
                    # Garder les bytes pour l'onglet d'identification
                    
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    with st.spinner(f"Traitement image {file_index + 1} avec params globaux..."):
                        processed_data = process_image(cv_image, tuned_params_for_all_images, 0)
                    
                    current_filtered_props = processed_data["filtered_props"]
                    num_detected_this_image = len(current_filtered_props)
                    grand_total_detected_insects += num_detected_this_image

                    # Stocker aussi les bytes de l'image originale pour l'onglet d'identification
                    all_images_results_temp.append({
                        "filename": uploaded_file.name,
                        "image_bytes": file_bytes, # Stocker les bytes
                        "cv_image_shape": cv_image.shape, # Pour reconstruire si nécessaire (ou stocker cv_image directement)
                        "results": processed_data,
                        "params_used_for_extraction": tuned_params_for_all_images.copy() # Stocker les params utilisés
                    })
                    
                    disp_col1, disp_col2 = st.columns(2)
                    with disp_col1:
                        st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption=f"Originale", width=300)
                    with disp_col2:
                        label_display = np.zeros((processed_data["labels"].shape[0], processed_data["labels"].shape[1], 3), dtype=np.uint8)
                        for prop_item in current_filtered_props:
                            color = np.random.randint(0, 255, size=3)
                            for coord in prop_item.coords:
                                label_display[coord[0], coord[1]] = color
                        st.image(label_display, caption=f"Insectes détectés: {num_detected_this_image} (Aire Min Globale: {tuned_params_for_all_images['min_area']})", use_column_width=True)

                    stat_col1, stat_col2 = st.columns(2)
                    stat_col1.metric(f"Insectes détectés (Image {file_index+1})", num_detected_this_image)
                    if current_filtered_props:
                        areas_this_image = [prop.area for prop in current_filtered_props]
                        stat_col2.metric(f"Surface moyenne (Image {file_index+1}, px)", f"{int(np.mean(areas_this_image)) if areas_this_image else 0}")
                    else:
                        stat_col2.metric(f"Surface moyenne (Image {file_index+1}, px)", "N/A")
                    
                    if current_filtered_props and num_detected_this_image > 0 :
                        # Pour l'extraction, il faut l'image cv2, pas les bytes.
                        # On la décode à nouveau si on ne l'a pas gardée.
                        nparr_preview = np.frombuffer(file_bytes, np.uint8)
                        cv_image_preview = cv2.imdecode(nparr_preview, cv2.IMREAD_COLOR)
                        extracted_preview = extract_insects(cv_image_preview, current_filtered_props[:min(3, num_detected_this_image)], tuned_params_for_all_images["margin"])
                        if extracted_preview:
                            st.write("Aperçu des 1ers insectes extraits:")
                            preview_cols_ext = st.columns(len(extracted_preview))
                            for i_ext, insect_ext_data in enumerate(extracted_preview):
                                preview_cols_ext[i_ext].image(cv2.cvtColor(insect_ext_data["image"], cv2.COLOR_BGR2RGB), width=100)
                    st.markdown("---")

                st.session_state.all_images_results = all_images_results_temp # Sauvegarder dans session_state
                st.session_state.grand_total_detected_insects = grand_total_detected_insects
                st.session_state.expected_insects_grand_total = expected_insects_grand_total
                st.session_state.segmentation_done = True # Marquer que la segmentation est faite
                st.rerun() # Forcer un rechargement pour afficher les résultats ci-dessous immédiatement

            # --- Affichage des résultats globaux après segmentation (si faite) ---
            if st.session_state.get('segmentation_done', False):
                st.header("Résultats Globaux Finaux de la Segmentation")
                expected_total = st.session_state.get('expected_insects_grand_total',0)
                detected_total = st.session_state.get('grand_total_detected_insects',0)

                st.metric("Nombre TOTAL d'insectes attendus (toutes images)", expected_total)
                st.metric("Nombre TOTAL d'insectes détectés (toutes images)", detected_total)
                
                diff_grand_total = abs(detected_total - expected_total)
                if diff_grand_total == 0:
                    st.success("✅ Succès! Nombre total d'insectes détectés correspond au nombre attendu.")
                elif diff_grand_total <= 0.1 * expected_total :
                    st.warning(f"⚠️ Attention: {detected_total} insectes détectés au total (attendu: {expected_total}, écart de {diff_grand_total}).")
                else:
                    st.error(f"❌ Erreur: {detected_total} insectes détectés au total (attendu: {expected_total}, écart important de {diff_grand_total}).")

                if st.button("Télécharger TOUS les insectes isolés (carrés, fond blanc)"):
                    temp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(temp_dir, "insectes_isoles_batch.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for result_item in st.session_state.all_images_results:
                            # Recharger l'image CV2 à partir des bytes stockés
                            nparr_zip = np.frombuffer(result_item["image_bytes"], np.uint8)
                            image_orig_for_zip = cv2.imdecode(nparr_zip, cv2.IMREAD_COLOR)
                            
                            filename_base = os.path.splitext(result_item["filename"])[0]
                            props_for_extraction = result_item["results"]["filtered_props"]
                            margin_for_extraction = result_item["params_used_for_extraction"]["margin"]
                            
                            extracted_insects_for_zip = extract_insects(image_orig_for_zip, props_for_extraction, margin_for_extraction)
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
            else:
                st.info("Veuillez téléverser des images et configurer les paramètres pour lancer la segmentation.")

    with tab2: # Onglet Identification
        st.header("Phase 2 : Identification des insectes")
        if model is None or class_names is None:
            st.error("Modèle d'identification non disponible. Impossible de procéder.")
        elif not st.session_state.get('segmentation_done', False) or not st.session_state.get('all_images_results'):
            st.info("Veuillez d'abord effectuer la segmentation dans l'onglet 'Segmentation'.")
        else:
            st.info(f"Modèle d'identification chargé. Classes: {', '.join(class_names)}")
            st.write("Les insectes détectés dans la phase de segmentation vont être identifiés.")

            # Un expander par image originale
            for i, result_item in enumerate(st.session_state.all_images_results):
                st.markdown(f"---")
                st.subheader(f"Identification pour l'image : {result_item['filename']}")

                # Reconstruire l'image cv2 à partir des bytes stockés
                nparr_ident = np.frombuffer(result_item["image_bytes"], np.uint8)
                cv_image_orig_ident = cv2.imdecode(nparr_ident, cv2.IMREAD_COLOR)

                props_for_extraction = result_item["results"]["filtered_props"]
                margin_for_extraction = result_item["params_used_for_extraction"]["margin"]
                
                # Extraire les insectes pour cette image (comme pour le zip)
                extracted_insects_for_id = extract_insects(cv_image_orig_ident, props_for_extraction, margin_for_extraction)

                if not extracted_insects_for_id:
                    st.write("Aucun insecte à identifier pour cette image.")
                    continue

                num_cols = 3 # Afficher 3 insectes par ligne
                cols = st.columns(num_cols)
                col_idx = 0

                for insect_data in extracted_insects_for_id:
                    insect_img_square_cv2 = insect_data["image"] # Ceci est une image cv2 BGR

                    # Prédiction
                    label, confidence, all_scores = predict_insect(insect_img_square_cv2, model, class_names, MODEL_INPUT_SIZE)
                    
                    current_col = cols[col_idx % num_cols]
                    with current_col:
                        st.image(cv2.cvtColor(insect_img_square_cv2, cv2.COLOR_BGR2RGB), 
                                 caption=f"Insecte #{insect_data['index'] + 1}", 
                                 width=150) # Taille d'affichage
                        if label == "Erreur Modèle":
                            st.error("Erreur de prédiction.")
                        else:
                            st.markdown(f"**Label:** {label}")
                            st.markdown(f"**Confiance:** {confidence*100:.2f}%")
                            
                            # Optionnel : afficher un petit graphique des scores
                            # import pandas as pd
                            # score_data = {"Classe": class_names, "Probabilité": all_scores}
                            # score_df = pd.DataFrame(score_data).sort_values(by="Probabilité", ascending=False)
                            # st.bar_chart(score_df.set_index("Classe").head(3)) # Top 3
                    col_idx += 1

    with tab3: # Onglet Guide
        st.header("Guide dʼoptimisation des paramètres")
        st.subheader("Approche Globale de Traitement")
        st.write("""
        Cette version de l'application adopte une approche de **paramètres globaux**. 
        - Les paramètres de détection (Flou, Constante C, Aire Minimale, etc.) sont déterminés **une seule fois** au début du traitement de segmentation.
        - Si les options "Auto-ajustement C/Aire Global" ou "Auto-ajustement du Flou Global" sont activées, ces paramètres sont optimisés en se basant sur la **première image** de votre lot. La cible pour cette optimisation est le `Nombre total dʼinsectes attendus / nombre d'images`.
        - Une fois déterminés, ces paramètres sont appliqués **identiquement à toutes les images** du lot pour la segmentation.
        - Le `Nombre total dʼinsectes attendus` que vous spécifiez est pour l'ensemble des images.
        """)

        st.subheader("Identification des Insectes")
        st.write(f"""
        Après la segmentation, l'onglet "Identification" utilise un modèle de Deep Learning (Keras) pour classifier chaque insecte isolé.
        - Le modèle attend des images carrées de taille {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]} pixels. Les images extraites sont automatiquement redimensionnées.
        - Les labels possibles sont : {', '.join(class_names) if class_names else "Non chargés"}.
        - Pour chaque insecte, l'application affiche le label prédit et le pourcentage de confiance.
        """)

        st.subheader("Filtrage Relatif des Petits Objets (Anti-Saleté)")
        st.write("""
        Pour réduire la détection de petites saletés lors de la segmentation :
        1. Après une détection initiale sur une image, si plusieurs objets sont trouvés, leur surface moyenne est calculée.
        2. Tout objet dont la surface est inférieure à 10% de cette moyenne est automatiquement écarté.
        3. Ce filtre est appliqué **par image**, car la "taille moyenne" est relative aux insectes de cette image spécifique. Il s'ajoute au filtre global `Surface minimale Aire`.
        """)
        # ... (reste du guide)


if __name__ == "__main__":
    # Initialiser st.session_state si ce n'est pas déjà fait
    if 'segmentation_done' not in st.session_state:
        st.session_state.segmentation_done = False
    if 'all_images_results' not in st.session_state:
        st.session_state.all_images_results = []
    if 'grand_total_detected_insects' not in st.session_state:
        st.session_state.grand_total_detected_insects = 0
    if 'expected_insects_grand_total' not in st.session_state:
        st.session_state.expected_insects_grand_total = 0
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = []
    main()

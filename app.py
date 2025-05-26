import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import clear_border
import os
import io
from PIL import Image, ImageOps
import tempfile
import zipfile
import base64
import tensorflow as tf
import matplotlib.pyplot as plt # Ajout pour le pie chart

# --- Configuration du Modèle d'Identification (SavedModel) ---
SAVED_MODEL_DIR_PATH = "model.savedmodel"
LABELS_PATH = "labels.txt"
MODEL_INPUT_SIZE = (224, 224)

# --- Fonctions de traitement d'image ---
def make_square(image, fill_color=(255, 255, 255)):
    height, width = image.shape[:2]
    max_side = max(height, width)
    top = (max_side - height) // 2
    bottom = max_side - height - top
    left = (max_side - width) // 2
    right = max_side - width - left
    square_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    return square_image

def process_image(image, params, expected_insects_for_image_info=0):
    blur_kernel = params["blur_kernel"]
    adapt_block_size = params["adapt_block_size"] # Doit être impair
    adapt_c = params["adapt_c"]
    min_area = params["min_area"]
    morph_kernel_size = params["morph_kernel"] # Doit être impair
    morph_iterations = params["morph_iterations"]
    auto_adjust_c_area_internally = params.get("auto_adjust_for_internal_tune", False)
    use_circularity = params.get("use_circularity", False)
    min_circularity = params.get("min_circularity", 0.3)
    apply_relative_filter = params.get("apply_relative_filter", True)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Flou gaussien
    if blur_kernel > 0: # Noyau doit être > 0
        blur_kernel_odd = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1 # Assurer impair
        blurred_img = cv2.GaussianBlur(gray, (blur_kernel_odd, blur_kernel_odd), 0)
    else:
        blurred_img = gray.copy() # Utiliser une copie pour éviter de modifier l'original 'gray'
        
    current_adapt_c = adapt_c
    current_min_area = min_area

    # S'assurer que adapt_block_size est impair et > 1
    adapt_block_size_odd = adapt_block_size if adapt_block_size % 2 != 0 else adapt_block_size + 1
    if adapt_block_size_odd <= 1: adapt_block_size_odd = 3
    
    # S'assurer que morph_kernel_size est impair
    morph_kernel_odd = morph_kernel_size if morph_kernel_size % 2 != 0 else morph_kernel_size + 1
    if morph_kernel_odd < 1: morph_kernel_odd = 1


    if auto_adjust_c_area_internally:
        # ... (logique d'auto-ajustement, peut nécessiter une révision si la segmentation de base est modifiée)
        # Pour l'instant, on se concentre sur la branche 'else' pour la segmentation manuelle
        # Cette partie de l'auto-ajustement devrait idéalement refléter la chaîne de traitement standard.
        # On va simplifier et supposer qu'elle utilise le même pipeline que 'else' pour la cohérence.
        adapt_c_values = [-5, 0, 2, 5, 8, 10, 15]
        best_params_auto = {"adapt_c": current_adapt_c}
        best_count_diff = float('inf')

        for ac_auto in adapt_c_values:
            thresh_auto = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block_size_odd, ac_auto)
            
            # Appliquer des opérations morphologiques similaires à la branche 'else'
            kernel_closing_auto = np.ones((morph_kernel_odd, morph_kernel_odd), np.uint8)
            closed_auto = cv2.morphologyEx(thresh_auto, cv2.MORPH_CLOSE, kernel_closing_auto, iterations=morph_iterations)
            
            kernel_opening_auto = np.ones((morph_kernel_odd, morph_kernel_odd), np.uint8)
            opened_auto = cv2.morphologyEx(closed_auto, cv2.MORPH_OPEN, kernel_opening_auto, iterations=1) # Moins d'itérations pour l'opening ici?

            cleared_auto = clear_border(opened_auto)
            labels_auto = measure.label(cleared_auto)
            props_auto = measure.regionprops(labels_auto)
            current_filtered_props_auto = [prop for prop in props_auto if prop.area >= current_min_area]
            count_diff = abs(len(current_filtered_props_auto) - expected_insects_for_image_info)
            if count_diff < best_count_diff:
                best_count_diff = count_diff
                best_params_auto["adapt_c"] = ac_auto
                if best_count_diff == 0: break
        current_adapt_c = best_params_auto["adapt_c"]

        # Retraiter avec le meilleur C trouvé
        thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block_size_odd, current_adapt_c)
        kernel_closing = np.ones((morph_kernel_odd, morph_kernel_odd), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_closing, iterations=morph_iterations)
        kernel_opening = np.ones((morph_kernel_odd, morph_kernel_odd), np.uint8)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening, iterations=1) # Une seule itération d'opening peut être plus sûre
        cleared = clear_border(opening)
        labels = measure.label(cleared)
        props = measure.regionprops(labels)
        filtered_props = [prop for prop in props if prop.area >= current_min_area]

    else: # Traitement standard
        thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block_size_odd, current_adapt_c)
        
        # MODIFICATION : Renforcer le CLOSING pour connecter les morceaux d'insectes
        # Un noyau plus grand et/ou plus d'itérations pour le closing.
        # Le noyau utilisé est `morph_kernel_odd` avec `morph_iterations`.
        # Vous pouvez expérimenter avec des itérations de closing plus élevées.
        kernel_closing = np.ones((morph_kernel_odd, morph_kernel_odd), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_closing, iterations=morph_iterations) # Utiliser les paramètres
        
        # MODIFICATION : OPENING pour enlever le bruit.
        # S'il est trop fort, il peut fragmenter. On utilise les paramètres, mais on pourrait
        # envisager 1 itération fixe pour l'opening si `morph_iterations` est élevé pour le closing.
        kernel_opening = np.ones((morph_kernel_odd, morph_kernel_odd), np.uint8)
        # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening, iterations=morph_iterations) # Si on veut le même nb d'itérations
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening, iterations=max(1, morph_iterations // 2)) # Moins d'opening que de closing

        cleared = clear_border(opening)
        labels = measure.label(cleared)
        props = measure.regionprops(labels)
        
        pre_filter_props = [prop for prop in props if prop.area >= current_min_area]

        if use_circularity:
            filtered_props_circ = []
            for prop in pre_filter_props:
                perimeter = prop.perimeter
                if perimeter > 0:
                    circularity = 4 * np.pi * prop.area / (perimeter * perimeter)
                    if circularity >= min_circularity:
                        filtered_props_circ.append(prop)
            filtered_props = filtered_props_circ
        else:
            filtered_props = pre_filter_props

    if apply_relative_filter and len(filtered_props) > 1:
        areas = [prop.area for prop in filtered_props]
        if areas: 
            avg_area = np.mean(areas)
            if avg_area > max(1.5 * current_min_area, 50):
                relative_threshold_area = 0.1 * avg_area
                final_relative_threshold = max(relative_threshold_area, current_min_area)
                final_filtered_props_after_relative = []
                for prop in filtered_props:
                    if prop.area >= final_relative_threshold :
                        final_filtered_props_after_relative.append(prop)
                filtered_props = final_filtered_props_after_relative

    final_params_used = params.copy()
    final_params_used['adapt_c'] = current_adapt_c
    final_params_used['min_area'] = current_min_area
    final_params_used['blur_kernel'] = blur_kernel
    return {"blurred": blurred_img, "thresh": thresh, "opening": opening, "labels": labels, # Retourner 'opening' (ou 'cleared')
            "filtered_props": filtered_props, "params_used": final_params_used}

# ... (extract_insects, load_saved_model_and_labels, predict_insect_saved_model, create_label_display_image restent les mêmes)
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
        extracted_insects.append({"image": square_insect, "index": i, "original_prop": prop})
    return extracted_insects

@st.cache_resource
def load_saved_model_and_labels(model_dir_path, labels_path):
    model_layer = None
    class_names_loaded = None
    try:
        abs_model_path = os.path.abspath(model_dir_path)
        if not os.path.exists(abs_model_path):
            print(f"DEBUG: Le dossier du modèle '{abs_model_path}' n'existe PAS.")
            return None, None
        if not os.path.isdir(abs_model_path):
            print(f"DEBUG: '{abs_model_path}' n'est PAS un dossier.")
            return None, None
        if not os.path.exists(os.path.join(abs_model_path, "saved_model.pb")):
            print(f"DEBUG: Le fichier 'saved_model.pb' est manquant dans '{abs_model_path}'.")
            return None, None

        call_endpoint_name = 'serving_default'
        try:
            model_layer = tf.keras.layers.TFSMLayer(abs_model_path, call_endpoint=call_endpoint_name)
        except Exception as e_tfsmlayer:
            print(f"DEBUG: Erreur TFSMLayer: {e_tfsmlayer}")
            try:
                loaded_sm = tf.saved_model.load(abs_model_path)
                available_signatures = list(loaded_sm.signatures.keys())
                print(f"DEBUG: Signatures disponibles: {available_signatures}")
            except Exception as e_load_sm:
                print(f"DEBUG: Erreur chargement SavedModel pour inspection: {e_load_sm}")
            return None, None

        abs_labels_path = os.path.abspath(labels_path)
        if not os.path.exists(abs_labels_path):
            print(f"DEBUG: Le fichier de labels '{abs_labels_path}' n'existe PAS.")
            return model_layer, None

        with open(abs_labels_path, "r") as f:
            class_names_loaded = [line.strip().split(" ", 1)[1] if " " in line.strip() else line.strip() for line in f.readlines()]
        return model_layer, class_names_loaded

    except Exception as e:
        print(f"DEBUG: Erreur générale chargement modèle/labels: {e}")
        return model_layer, class_names_loaded

def predict_insect_saved_model(image_cv2, model_layer, class_names, input_size):
    if model_layer is None:
        return "Erreur Modèle Non Chargé", 0.0, []
    if class_names is None:
        return "Erreur Labels Non Chargés", 0.0, []

    img_resized = cv2.resize(image_cv2, input_size, interpolation=cv2.INTER_AREA)
    image_array = np.asarray(img_resized, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1.0
    input_tensor = tf.convert_to_tensor(normalized_image_array)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    predictions_np = None
    try:
        predictions_output = model_layer(input_tensor)
        
        if isinstance(predictions_output, dict):
            if len(predictions_output) == 1:
                predictions_tensor = list(predictions_output.values())[0]
            elif 'outputs' in predictions_output:
                 predictions_tensor = predictions_output['outputs']
            elif 'output_0' in predictions_output:
                 predictions_tensor = predictions_output['output_0']
            else:
                key_found = None
                for key, value in predictions_output.items():
                    if isinstance(value, tf.Tensor) and len(value.shape) == 2 and value.shape[0] == 1:
                        predictions_tensor = value
                        key_found = key
                        break
                if key_found is None:
                    return "Erreur Sortie Modèle Dict", 0.0, []
        else:
            predictions_tensor = predictions_output

        if hasattr(predictions_tensor, 'numpy'):
            predictions_np = predictions_tensor.numpy()
        else:
            predictions_np = np.array(predictions_tensor)

    except Exception as e_predict:
        print(f"DEBUG: Erreur prédiction: {e_predict}")
        return "Erreur Prédiction", 0.0, []

    if predictions_np is None or predictions_np.size == 0:
        return "Erreur Prédiction Vide", 0.0, []

    predicted_class_index = np.argmax(predictions_np[0])
    confidence_score = predictions_np[0][predicted_class_index]
    
    if predicted_class_index >= len(class_names):
        return "Erreur Index Label", confidence_score, predictions_np[0]
            
    label_name = class_names[predicted_class_index]
    
    return label_name, confidence_score, predictions_np[0]

def create_label_display_image(label_image_data, filtered_props):
    if label_image_data.ndim == 3 and label_image_data.shape[2] == 1:
        label_image_data = label_image_data.squeeze(axis=2)
    elif label_image_data.ndim != 2:
        h, w = (200, 200) if not filtered_props or not hasattr(filtered_props[0], 'image') else filtered_props[0].image.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    label_display = np.zeros((label_image_data.shape[0], label_image_data.shape[1], 3), dtype=np.uint8)
    for prop_item in filtered_props:
        color = np.random.randint(50, 256, size=3)
        for coord in prop_item.coords:
            if 0 <= coord[0] < label_display.shape[0] and 0 <= coord[1] < label_display.shape[1]:
                label_display[coord[0], coord[1]] = color
    return label_display


ECOLOGICAL_FUNCTIONS_MAP = {
    "Apidae": "Pollinisateurs",
    "Isopodes": "Décomposeurs",
    "Carabide": "Prédateurs",      # Note: "Carabide" et non "Carabes". Si votre modèle prédit "Carabidae", il faudra ajuster.
    "Arachnides": "Prédateurs",
    "Mouches des semis": "Ravageur"
}
DEFAULT_ECOLOGICAL_FUNCTION = "Non défini" # Pour tout autre label non listé ci-dessus

def main():
    st.title("Détection, isolation et identification dʼinsectes")
    # ... (le reste de main() jusqu'à la sidebar)

    model, class_names = load_saved_model_and_labels(SAVED_MODEL_DIR_PATH, LABELS_PATH)

    if model is None:
        st.warning("Le modèle d'identification (SavedModel) n'a pas pu être chargé.")
    if class_names is None and model is not None:
        st.warning("Le fichier de labels n'a pas pu être chargé.")

    with st.sidebar:
        st.header("Paramètres de détection Globaux")
        
        default_expected_insects = len(st.session_state.get('uploaded_files_main_cache', [])) * 3 if st.session_state.get('uploaded_files_main_cache') else 3
        expected_insects_grand_total = st.number_input(
            "Nombre total dʼinsectes attendus (toutes images)", 
            min_value=1, 
            value=st.session_state.get('expected_insects_grand_total_val', default_expected_insects), 
            step=1, 
            key='expected_insects_grand_total_key'
        )
        st.session_state.expected_insects_grand_total_val = expected_insects_grand_total

        presets = {
            "Par défaut": {"blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 5, "morph_kernel": 3, "morph_iterations": 2, "min_area": 100, "margin": 15, "apply_relative_filter": True},
            "Grands insectes": {"blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 8, "morph_kernel": 5, "morph_iterations": 2, "min_area": 500, "margin": 15, "apply_relative_filter": True},
            "Petits insectes (bruité)": {"blur_kernel": 5, "adapt_block_size": 25, "adapt_c": 2, "morph_kernel": 5, "morph_iterations": 3, "min_area": 50, "margin": 10, "apply_relative_filter": False},
        }
        preset_choice_options = ["Personnalisé", "Auto-ajustement C/Aire Global"] + list(presets.keys())
        try:
            default_preset_index = preset_choice_options.index(st.session_state.get('preset_choice_val', "Par défaut"))
        except ValueError:
            default_preset_index = preset_choice_options.index("Par défaut") if "Par défaut" in preset_choice_options else 0
        preset_choice = st.selectbox(
            "Configurations prédéfinies", 
            preset_choice_options, 
            index=default_preset_index, 
            key='preset_choice_key'
        )
        st.session_state.preset_choice_val = preset_choice

        base_params_config = presets["Par défaut"].copy()
        def get_initial_param_value(param_name, default_value_map, chosen_preset, global_config_map, session_key_suffix='_val'):
            session_key = f'{param_name}{session_key_suffix}'
            if chosen_preset in default_value_map:
                preset_value = default_value_map[chosen_preset].get(param_name)
                if preset_value is not None:
                    st.session_state[session_key] = preset_value
                    return preset_value
            return st.session_state.get(session_key, global_config_map.get(param_name, 0))

        current_params_config = {}
        # MODIFICATION : Pas de 1 pour le flou gaussien
        current_params_config["blur_kernel"] = st.slider("Noyau de flou gaussien (0 pour désactiver, sinon impair)", 0, 21, get_initial_param_value("blur_kernel", presets, preset_choice, base_params_config), step=1, key="blur_glob_key_slider")
        st.session_state.blur_kernel_val = current_params_config["blur_kernel"]

        current_params_config["adapt_block_size"] = st.slider("Taille du bloc adaptatif (impair, >1)", 3, 51, get_initial_param_value("adapt_block_size", presets, preset_choice, base_params_config), step=2, key="block_glob_key_slider")
        st.session_state.adapt_block_size_val = current_params_config["adapt_block_size"]
        
        current_params_config["adapt_c"] = st.slider("Constante de seuillage C", -20, 20, get_initial_param_value("adapt_c", presets, preset_choice, base_params_config), step=1, key="c_glob_key_slider") # Plage ajustée
        st.session_state.adapt_c_val = current_params_config["adapt_c"]

        current_params_config["min_area"] = st.slider("Surface minimale Aire (pixels)", 10, 10000, get_initial_param_value("min_area", presets, preset_choice, base_params_config), step=10, key="area_glob_key_slider")
        st.session_state.min_area_val = current_params_config["min_area"]

        current_params_config["morph_kernel"] = st.slider("Noyau morphologique (impair)", 1, 15, get_initial_param_value("morph_kernel", presets, preset_choice, base_params_config), step=2, key="morph_k_glob_key_slider") # Plage augmentée
        st.session_state.morph_kernel_val = current_params_config["morph_kernel"]
        
        current_params_config["morph_iterations"] = st.slider("Itérations morphologiques (Closing puis Opening)", 1, 5, get_initial_param_value("morph_iterations", presets, preset_choice, base_params_config), key="morph_i_glob_key_slider")
        st.session_state.morph_iterations_val = current_params_config["morph_iterations"]

        current_params_config["margin"] = st.slider("Marge autour des insectes", 0, 50, get_initial_param_value("margin", presets, preset_choice, base_params_config), key="margin_glob_key_slider")
        st.session_state.margin_val = current_params_config["margin"]
        
        current_params_config["use_circularity"] = st.checkbox("Filtrer par circularité", value=st.session_state.get('use_circularity_val', False), key="circularity_check_key_box")
        st.session_state.use_circularity_val = current_params_config["use_circularity"]

        if current_params_config["use_circularity"]:
            current_params_config["min_circularity"] = st.slider("Circularité minimale", 0.0, 1.0, st.session_state.get('min_circularity_val', 0.3), step=0.05, key="min_circularity_key_slider")
            st.session_state.min_circularity_val = current_params_config["min_circularity"]
        else:
            current_params_config["min_circularity"] = st.session_state.get('min_circularity_val', 0.3)

        should_auto_adjust_blur_globally = st.checkbox("Auto-ajustement du Flou Global (sur 1ère image)", value=st.session_state.get('auto_adjust_blur_val', False), key="auto_blur_key_box")
        st.session_state.auto_adjust_blur_val = should_auto_adjust_blur_globally
        
        current_params_config["apply_relative_filter"] = st.checkbox("Appliquer filtre de taille relative (anti-saleté)", value=get_initial_param_value("apply_relative_filter", presets, preset_choice, base_params_config, session_key_suffix='_rel_filter_val'), key="relative_filter_key_box")
        st.session_state.apply_relative_filter_rel_filter_val = current_params_config["apply_relative_filter"]

        st.session_state.current_processing_params = current_params_config
        st.session_state.current_preset_choice_for_logic = preset_choice
        st.session_state.current_should_auto_adjust_blur_for_logic = should_auto_adjust_blur_globally
    
    # --- Onglets principaux ---
    tab1, tab2, tab3 = st.tabs(["Segmentation", "Identification", "Guide dʼutilisation"])

    with tab1:
        # ... (Logique de l'onglet Segmentation comme dans la version précédente)
        # Assurez-vous que `processed_data["opening"]` ou `processed_data["cleared"]`
        # est bien l'image que vous voulez afficher comme étape intermédiaire de segmentation.
        # Actuellement, `create_label_display_image` utilise `processed_data["labels"]`
        # et `filtered_props` pour colorer les régions finales.
        # Si vous voulez voir l'image binaire *avant* le label_display, il faut la stocker et l'afficher.
        # Par exemple, dans la boucle d'affichage des résultats :
        #   st.image(result_item["processed_data"]["opening"], caption="Résultat Morphologique (Opening)", use_column_width=True)
        # Et s'assurer que "opening" (ou l'étape que vous voulez) est bien retournée par `process_image` et stockée.
        st.header("Phase 1 : Détection et Segmentation des insectes")
        uploaded_files_main = st.file_uploader(
            "Choisissez une ou plusieurs images pour la segmentation", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True, 
            key="file_uploader_main_tab1"
        )
        st.session_state.uploaded_files_main_cache = uploaded_files_main

        if uploaded_files_main:
            current_file_names = [f.name for f in uploaded_files_main]
            params_changed = False
            if 'last_used_processing_params' in st.session_state:
                if st.session_state.last_used_processing_params != st.session_state.current_processing_params:
                    params_changed = True
            else:
                params_changed = True

            if st.session_state.get('last_processed_file_names') != current_file_names or \
               not st.session_state.get('segmentation_done', False) or params_changed:

                st.session_state.segmentation_done = False 
                st.session_state.all_images_results = [] 
                params_to_use = st.session_state.current_processing_params
                active_preset_choice = st.session_state.current_preset_choice_for_logic
                active_should_auto_adjust_blur = st.session_state.current_should_auto_adjust_blur_for_logic
                active_expected_total_insects = st.session_state.expected_insects_grand_total_val
                st.session_state.last_used_processing_params = params_to_use.copy()

                all_images_results_temp = []
                grand_total_detected_insects = 0
                
                st.markdown("---")
                st.subheader("Détermination des Paramètres Globaux (basée sur la 1ère image)")
                tuned_params_for_all_images = params_to_use.copy()

                first_image_obj = uploaded_files_main[0]
                # ... (reste de la logique d'auto-ajustement et de traitement des images) ...
                first_image_bytes = first_image_obj.getvalue()
                first_nparr = np.frombuffer(first_image_bytes, np.uint8)
                first_cv_image = cv2.imdecode(first_nparr, cv2.IMREAD_COLOR)
                expected_for_first_image_tuning = max(1, round(active_expected_total_insects / len(uploaded_files_main)))
                st.write(f"Image de référence pour l'ajustement global: {first_image_obj.name} (cible pour cette image: ~{expected_for_first_image_tuning} insectes)")

                should_auto_adjust_c_area_logic = (active_preset_choice == "Auto-ajustement C/Aire Global")

                if should_auto_adjust_c_area_logic:
                    with st.spinner("Ajustement global C/Aire en cours..."):
                        params_for_c_area_tune = tuned_params_for_all_images.copy()
                        params_for_c_area_tune["min_area"] = tuned_params_for_all_images["min_area"]
                        params_for_c_area_tune["auto_adjust_for_internal_tune"] = True
                        c_area_tune_data = process_image(first_cv_image, params_for_c_area_tune, expected_for_first_image_tuning)
                        tuned_params_for_all_images["adapt_c"] = c_area_tune_data["params_used"]["adapt_c"]
                    st.success(f"Global C/Aire déterminés: C={tuned_params_for_all_images['adapt_c']}, Aire Min (utilisée): {tuned_params_for_all_images['min_area']}")


                if active_should_auto_adjust_blur:
                    with st.spinner("Ajustement global du Flou en cours..."):
                        blur_kernel_options = [k for k in range(1, 22, 2) if k > 0] # Noyau de flou > 0
                        if tuned_params_for_all_images["blur_kernel"] == 0 and blur_kernel_options: # Si flou désactivé, commencer par le plus petit
                            best_blur_found_globally = blur_kernel_options[0]
                        else:
                            best_blur_found_globally = tuned_params_for_all_images["blur_kernel"]
                        
                        min_diff_for_global_blur = float('inf')
                        c_after_blur_tune = tuned_params_for_all_images["adapt_c"]
                        # area_after_blur_tune = tuned_params_for_all_images["min_area"]
                        
                        # Tester aussi sans flou si le flou actuel n'est pas 0
                        if tuned_params_for_all_images["blur_kernel"] != 0:
                            blur_kernel_options_with_zero = [0] + blur_kernel_options
                        else:
                            blur_kernel_options_with_zero = blur_kernel_options


                        for trial_blur in blur_kernel_options_with_zero:
                            params_for_blur_trial = tuned_params_for_all_images.copy()
                            params_for_blur_trial["blur_kernel"] = trial_blur
                            params_for_blur_trial["auto_adjust_for_internal_tune"] = should_auto_adjust_c_area_logic
                            trial_data = process_image(first_cv_image, params_for_blur_trial, expected_for_first_image_tuning)
                            trial_num_detected = len(trial_data["filtered_props"])
                            trial_diff = abs(trial_num_detected - expected_for_first_image_tuning)
                            if trial_diff < min_diff_for_global_blur:
                                min_diff_for_global_blur = trial_diff
                                best_blur_found_globally = trial_blur
                                if should_auto_adjust_c_area_logic:
                                    c_after_blur_tune = trial_data["params_used"]["adapt_c"]
                            elif trial_diff == min_diff_for_global_blur and trial_blur < best_blur_found_globally: # Préférer un flou plus petit si égalité
                                best_blur_found_globally = trial_blur
                                if should_auto_adjust_c_area_logic:
                                    c_after_blur_tune = trial_data["params_used"]["adapt_c"]
                            if min_diff_for_global_blur == 0: break
                        tuned_params_for_all_images["blur_kernel"] = best_blur_found_globally
                        if should_auto_adjust_c_area_logic:
                            tuned_params_for_all_images["adapt_c"] = c_after_blur_tune
                            st.success(f"Global Flou déterminé: {tuned_params_for_all_images['blur_kernel']}. Global C/Aire (après Flou): C={tuned_params_for_all_images['adapt_c']}, Aire Min (utilisée): {tuned_params_for_all_images['min_area']}")
                        else:
                            st.success(f"Global Flou déterminé: {tuned_params_for_all_images['blur_kernel']}.")
                
                tuned_params_for_all_images["auto_adjust_for_internal_tune"] = False
                st.session_state.tuned_params_for_all_images = tuned_params_for_all_images
                
                st.markdown("---")
                st.subheader("Traitement des Images avec Paramètres Globaux")
                st.json(st.session_state.tuned_params_for_all_images)

                with st.spinner("Traitement de toutes les images..."):
                    for file_index, uploaded_file in enumerate(uploaded_files_main):
                        file_bytes = uploaded_file.getvalue()
                        nparr = np.frombuffer(file_bytes, np.uint8)
                        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        processed_data = process_image(cv_image, st.session_state.tuned_params_for_all_images, 0)
                        current_filtered_props = processed_data["filtered_props"]
                        num_detected_this_image = len(current_filtered_props)
                        grand_total_detected_insects += num_detected_this_image
                        all_images_results_temp.append({
                            "filename": uploaded_file.name, "image_bytes": file_bytes,
                            "cv_image_color": cv_image, "processed_data": processed_data,
                            "num_detected": num_detected_this_image,
                            "params_used_for_extraction": st.session_state.tuned_params_for_all_images.copy()
                        })
                st.session_state.all_images_results = all_images_results_temp
                st.session_state.grand_total_detected_insects = grand_total_detected_insects
                st.session_state.expected_insects_grand_total_final = active_expected_total_insects
                st.session_state.segmentation_done = True
                st.session_state.last_processed_file_names = current_file_names
                st.rerun()

        if st.session_state.get('segmentation_done', False) and st.session_state.get('all_images_results'):
            st.header("Résultats de la Segmentation par Image")
            for idx, result_item in enumerate(st.session_state.all_images_results):
                st.markdown(f"#### Image {idx + 1}: {result_item['filename']}")
                
                # MODIFICATION : Afficher 3 images: Originale, Résultat Morpho, Labels colorés
                col1, col2, col3 = st.columns(3) 
                with col1:
                    st.image(cv2.cvtColor(result_item["cv_image_color"], cv2.COLOR_BGR2RGB), 
                             caption=f"Originale", 
                             use_column_width=True)
                with col2:
                    # Afficher l'image après les opérations morphologiques (ex: 'opening' ou 'cleared')
                    # S'assurer que 'opening' est bien une image binaire 0-255
                    morpho_result_img = result_item["processed_data"]["opening"] 
                    st.image(morpho_result_img, clamp=True, channels="GRAY", # clamp pour s'assurer des valeurs 0-255 si besoin
                             caption="Résultat Morphologique", 
                             use_column_width=True)
                with col3:
                    label_image_from_processing = result_item["processed_data"]["labels"]
                    filtered_props_for_display = result_item["processed_data"]["filtered_props"]
                    display_img_labels = create_label_display_image(label_image_from_processing, filtered_props_for_display)
                    st.image(display_img_labels, 
                             caption=f"Insectes Détectés: {result_item['num_detected']}", 
                             use_column_width=True)

                # ... (Stats et aperçu des extraits comme avant) ...
                stat_col1, stat_col2 = st.columns(2)
                stat_col1.metric(f"Insectes détectés (Image {idx+1})", result_item['num_detected'])
                if result_item["processed_data"]["filtered_props"]:
                    areas_this_image = [prop.area for prop in result_item["processed_data"]["filtered_props"]]
                    stat_col2.metric(f"Surface moyenne (Image {idx+1}, px)", f"{int(np.mean(areas_this_image)) if areas_this_image else 'N/A'}")
                else:
                    stat_col2.metric(f"Surface moyenne (Image {idx+1}, px)", "N/A")
                if result_item["processed_data"]["filtered_props"] and result_item['num_detected'] > 0 :
                    margin_for_preview = st.session_state.tuned_params_for_all_images["margin"]
                    extracted_preview = extract_insects(result_item["cv_image_color"], result_item["processed_data"]["filtered_props"][:min(3, result_item['num_detected'])], margin_for_preview)
                    if extracted_preview:
                        st.write("Aperçu des 1ers insectes extraits:")
                        preview_cols_ext = st.columns(len(extracted_preview))
                        for i_ext, insect_ext_data in enumerate(extracted_preview):
                            preview_cols_ext[i_ext].image(cv2.cvtColor(insect_ext_data["image"], cv2.COLOR_BGR2RGB), width=100)
                st.markdown("---")


            st.header("Résultats Globaux Finaux de la Segmentation")
            # ... (Affichage des résultats globaux et bouton de téléchargement) ...
            expected_total_final = st.session_state.get('expected_insects_grand_total_final',0)
            detected_total_final = st.session_state.get('grand_total_detected_insects',0)
            st.metric("Nombre TOTAL d'insectes attendus (toutes images)", expected_total_final)
            st.metric("Nombre TOTAL d'insectes détectés (toutes images)", detected_total_final)
            diff_grand_total = abs(detected_total_final - expected_total_final)
            if diff_grand_total == 0 and expected_total_final > 0 :
                st.success("✅ Succès! Nombre total d'insectes détectés correspond au nombre attendu.")
            elif expected_total_final > 0 and diff_grand_total <= 0.1 * expected_total_final :
                st.warning(f"⚠️ Attention: {detected_total_final} insectes détectés au total (attendu: {expected_total_final}, écart de {diff_grand_total}).")
            else:
                st.error(f"❌ Écart: {detected_total_final} insectes détectés au total (attendu: {expected_total_final}, écart de {diff_grand_total}).")

            if st.button("Télécharger TOUS les insectes isolés (carrés, fond blanc)", key="download_all_zip_tab1"):
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "insectes_isoles_batch.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for res_item_zip in st.session_state.all_images_results:
                        image_orig_for_zip = res_item_zip["cv_image_color"]
                        filename_base = os.path.splitext(res_item_zip["filename"])[0]
                        props_for_extraction = res_item_zip["processed_data"]["filtered_props"]
                        margin_for_extraction = st.session_state.tuned_params_for_all_images["margin"]
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


        elif not uploaded_files_main and st.session_state.get('segmentation_done', False):
            st.session_state.segmentation_done = False
            st.session_state.all_images_results = []
            st.session_state.last_processed_file_names = None
            st.info("Veuillez téléverser de nouvelles images pour la segmentation.")

        elif not uploaded_files_main :
            st.info("Veuillez téléverser des images pour commencer la segmentation.")


    with tab2: # Onglet Identification
        st.header("Phase 2 : Identification des insectes")
        if model is None or class_names is None: # Vérification simplifiée
            st.error("Modèle d'identification ou labels non disponibles. Impossible de procéder.")
        elif not st.session_state.get('segmentation_done', False) or not st.session_state.get('all_images_results'):
            st.info("Veuillez d'abord effectuer la segmentation dans l'onglet 'Segmentation'.")
        else:
            st.info(f"Modèle d'identification chargé. Classes: {len(class_names)}")
            
            all_identified_labels = []
            for result_item in st.session_state.all_images_results:
                # ... (Logique d'extraction et prédiction pour chaque image) ...
                if "cv_image_color" in result_item:
                    cv_image_orig_ident = result_item["cv_image_color"]
                else:
                    nparr_ident = np.frombuffer(result_item["image_bytes"], np.uint8)
                    cv_image_orig_ident = cv2.imdecode(nparr_ident, cv2.IMREAD_COLOR)
                props_for_extraction = result_item["processed_data"]["filtered_props"]
                margin_for_extraction = result_item.get("params_used_for_extraction",{}).get("margin", st.session_state.get("tuned_params_for_all_images",{}).get("margin",15) )
                extracted_insects_for_id = extract_insects(cv_image_orig_ident, props_for_extraction, margin_for_extraction)

                for insect_data in extracted_insects_for_id:
                    label, confidence, _ = predict_insect_saved_model(insect_data["image"], model, class_names, MODEL_INPUT_SIZE)
                    if "Erreur" not in label:
                        all_identified_labels.append(label)
            
            # Compter les fonctions écologiques
            ecological_counts = {}
            for identified_label in all_identified_labels:
                eco_function = ECOLOGICAL_FUNCTIONS_MAP.get(identified_label, DEFAULT_ECOLOGICAL_FUNCTION)
                ecological_counts[eco_function] = ecological_counts.get(eco_function, 0) + 1
            
            if ecological_counts:
                st.subheader("Répartition des Fonctions Écologiques")
                
                # Créer le pie chart avec Matplotlib
                labels_pie = list(ecological_counts.keys())
                sizes_pie = list(ecological_counts.values())
                
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal') # Assure que le pie est un cercle.
                st.pyplot(fig1)
            else:
                st.write("Aucun insecte n'a pu être identifié pour générer le graphique des fonctions écologiques.")

            # Affichage détaillé des insectes identifiés par image (comme avant)
            for i, result_item in enumerate(st.session_state.all_images_results):
                st.markdown(f"---")
                st.subheader(f"Détail identification pour : {result_item['filename']}")
                # ... (répéter la logique d'affichage des insectes identifiés par image) ...
                if "cv_image_color" in result_item:
                    cv_image_orig_ident = result_item["cv_image_color"]
                else:
                    nparr_ident = np.frombuffer(result_item["image_bytes"], np.uint8)
                    cv_image_orig_ident = cv2.imdecode(nparr_ident, cv2.IMREAD_COLOR)

                props_for_extraction = result_item["processed_data"]["filtered_props"]
                margin_for_extraction = result_item.get("params_used_for_extraction",{}).get("margin", st.session_state.get("tuned_params_for_all_images",{}).get("margin",15) )
                extracted_insects_for_id = extract_insects(cv_image_orig_ident, props_for_extraction, margin_for_extraction)

                if not extracted_insects_for_id:
                    st.write("Aucun insecte à identifier pour cette image.")
                    continue
                num_cols = 3
                cols = st.columns(num_cols)
                col_idx = 0
                for insect_data in extracted_insects_for_id:
                    insect_img_square_cv2 = insect_data["image"]
                    label, confidence, all_scores = predict_insect_saved_model(insect_img_square_cv2, model, class_names, MODEL_INPUT_SIZE)
                    current_col = cols[col_idx % num_cols]
                    with current_col:
                        st.image(cv2.cvtColor(insect_img_square_cv2, cv2.COLOR_BGR2RGB), caption=f"Insecte #{insect_data['index'] + 1}", width=150)
                        if "Erreur" in label:
                            st.error(f"{label} (Confiance: {confidence*100:.2f}%)")
                        else:
                            st.markdown(f"**Label:** {label}")
                            st.markdown(f"**Fonction:** {ECOLOGICAL_FUNCTIONS_MAP.get(label, DEFAULT_ECOLOGICAL_FUNCTION)}")
                            st.markdown(f"**Confiance:** {confidence*100:.2f}%")
                    col_idx += 1


    with tab3: # Onglet Guide
        # ... (Guide comme avant, mais ajouter une note sur le pie chart et les fonctions écologiques)
        st.header("Guide dʼoptimisation des paramètres")
        st.subheader("Identification des Insectes")
        st.write(f"""
        Après la segmentation, l'onglet "Identification" utilise un modèle TensorFlow SavedModel pour classifier chaque insecte isolé.
        - Le modèle attend des images carrées de taille {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]} pixels.
        - Les labels possibles sont : {', '.join(class_names) if class_names else "Labels non chargés"}.
        - Un graphique circulaire (camembert) montre la répartition des fonctions écologiques des insectes identifiés sur l'ensemble des images.
        """)
        st.subheader("Paramètres de Segmentation")
        st.write("""
        - **Noyau de flou gaussien**: Le pas est maintenant de 1 pour plus de précision. 0 désactive le flou. Un flou léger peut aider à réduire le bruit.
        - **Opérations morphologiques**: L'ordre est THRESH -> CLOSING -> OPENING.
            - *Closing* (avec `Noyau morphologique` et `Itérations morphologiques`) est appliqué pour remplir les trous et connecter les parties proches des objets.
            - *Opening* (avec `Noyau morphologique` et la moitié des itérations du closing, au moins 1) est appliqué ensuite pour enlever le bruit fin et lisser les contours. Si les insectes sont trop fragmentés, essayez d'augmenter les itérations de *Closing* ou de réduire légèrement celles de *Opening* (indirectement en augmentant celles du closing).
        - **Surface minimale Aire**: Principal filtre de taille. Augmentez pour ignorer les petits objets.
        """)


if __name__ == "__main__":
    # ... (Initialisation de st.session_state comme avant, s'assurer que les nouvelles clés sont là)
    keys_to_initialize = {
        'segmentation_done': False, 'all_images_results': [], 'grand_total_detected_insects': 0,
        'expected_insects_grand_total_val': 3, 'expected_insects_grand_total_final': 0,
        'last_processed_file_names': None, 'preset_choice_val': "Par défaut",
        'blur_kernel_val': 7, 'adapt_block_size_val': 35, 'adapt_c_val': 5,
        'min_area_val': 100, 'morph_kernel_val': 3, 'morph_iterations_val': 2,
        'margin_val': 15, 'use_circularity_val': False, 'min_circularity_val': 0.3,
        'auto_adjust_blur_val': False, 'apply_relative_filter_rel_filter_val': True,
        'tuned_params_for_all_images': {}, 'uploaded_files_main_cache': None,
        'current_processing_params': {}, 'current_preset_choice_for_logic': "Par défaut",
        'current_should_auto_adjust_blur_for_logic': False, 'last_used_processing_params': None
    }
    for key, default_value in keys_to_initialize.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    main()

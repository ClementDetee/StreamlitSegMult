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
import matplotlib.pyplot as plt
import math

# --- Configuration Globale ---
SAVED_MODEL_DIR_PATH = "model.savedmodel"
LABELS_PATH = "labels.txt"
MODEL_INPUT_SIZE = (224, 224)

ECOLOGICAL_FUNCTIONS_MAP = {
    "Apidae": "Pollinisateurs",
    "Isopodes": "Décomposeurs",
    "Carabide": "Prédateurs", # VÉRIFIEZ CE LABEL EXACTEMENT
    "Arachnides": "Prédateurs",
    "Mouches des semis": "Ravageur"
}
DEFAULT_ECOLOGICAL_FUNCTION = "Non défini"

DEFAULT_SEG_PARAMS = {
    "target_insect_count": 3,
    "blur_kernel": 5,
    "adapt_block_size": 35,
    "adapt_c": 5,
    "min_area": 150,
    "morph_kernel": 3,
    "morph_iterations": 2,
    "margin": 15,
    "use_circularity": False,
    "min_circularity": 0.3,
    "apply_relative_filter": True
}

# --- Fonctions Utilitaires ---
def make_square(image, fill_color=(255, 255, 255)):
    height, width = image.shape[:2]
    max_side = max(height, width)
    top = (max_side - height) // 2
    bottom = max_side - height - top
    left = (max_side - width) // 2
    right = max_side - width - left
    square_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    return square_image

def calculate_shannon_index(counts_dict):
    if not counts_dict or sum(counts_dict.values()) == 0:
        return 0.0
    total_individuals = sum(counts_dict.values())
    shannon_index = 0.0
    for category_count in counts_dict.values(): # Itérer sur les comptes directement
        if category_count > 0:
            proportion = category_count / total_individuals
            shannon_index -= proportion * math.log(proportion)
    return shannon_index

# --- Fonctions de Traitement d'Image et Modèle ---
def process_image(image_cv, params, target_insect_count_for_tune=0, auto_tune_mode=False):
    blur_kernel_orig = params["blur_kernel"]
    adapt_block_size_orig = params["adapt_block_size"]
    adapt_c_orig = params["adapt_c"]
    min_area_param_orig = params["min_area"]
    morph_kernel_size_orig = params["morph_kernel"]
    morph_iterations_orig = params["morph_iterations"]
    use_circularity = params.get("use_circularity", False)
    min_circularity = params.get("min_circularity", 0.3)
    apply_relative_filter = params.get("apply_relative_filter", True)

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    current_params_to_use = params.copy() # Commence avec les params donnés

    if auto_tune_mode and target_insect_count_for_tune > 0:
        # st.write(f"Auto-tuning pour {target_insect_count_for_tune} insectes...")
        blur_options = [0, 3, 5, 7, 9] 
        adapt_c_options = [-10, -5, 0, 2, 5, 8, 10]
        
        best_params_tuned = current_params_to_use.copy()
        best_diff = float('inf')
        processed_results_for_best_tune = None
        
        for test_blur in blur_options:
            for test_c in adapt_c_options:
                temp_params = current_params_to_use.copy()
                temp_params["blur_kernel"] = test_blur
                temp_params["adapt_c"] = test_c
                
                # Exécuter une passe de segmentation avec ces params de test
                # (logique de segmentation interne simplifiée pour le tuning)
                if temp_params["blur_kernel"] > 0:
                    b_k_odd = temp_params["blur_kernel"] if temp_params["blur_kernel"] % 2 != 0 else temp_params["blur_kernel"] + 1
                    blurred_tune = cv2.GaussianBlur(gray, (b_k_odd, b_k_odd), 0)
                else:
                    blurred_tune = gray.copy()

                a_b_s_odd = temp_params["adapt_block_size"] if temp_params["adapt_block_size"] % 2 != 0 else temp_params["adapt_block_size"] + 1
                if a_b_s_odd <= 1: a_b_s_odd = 3
                m_k_s_odd = temp_params["morph_kernel"] if temp_params["morph_kernel"] % 2 != 0 else temp_params["morph_kernel"] + 1
                if m_k_s_odd < 1: m_k_s_odd = 1

                thresh_tune = cv2.adaptiveThreshold(blurred_tune, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, a_b_s_odd, temp_params["adapt_c"])
                kernel_closing_tune = np.ones((m_k_s_odd, m_k_s_odd), np.uint8)
                closing_tune = cv2.morphologyEx(thresh_tune, cv2.MORPH_CLOSE, kernel_closing_tune, iterations=temp_params["morph_iterations"])
                kernel_opening_tune = np.ones((m_k_s_odd, m_k_s_odd), np.uint8)
                opening_tune = cv2.morphologyEx(closing_tune, cv2.MORPH_OPEN, kernel_opening_tune, iterations=max(1, temp_params["morph_iterations"] // 2))
                cleared_tune = clear_border(opening_tune)
                labels_tune = measure.label(cleared_tune)
                props_tune = measure.regionprops(labels_tune)
                filtered_props_tune = [p for p in props_tune if p.area >= temp_params["min_area"]]
                # Note: Pour un tuning complet, les filtres de circularité et relatif devraient aussi être appliqués ici

                num_detected_test = len(filtered_props_tune)
                current_diff = abs(num_detected_test - target_insect_count_for_tune)

                if current_diff < best_diff:
                    best_diff = current_diff
                    best_params_tuned = temp_params.copy()
                    # Stocker les résultats intermédiaires du meilleur tune pour ne pas avoir à recalculer
                    processed_results_for_best_tune = {
                        "blurred": blurred_tune, "thresh": thresh_tune, "opening": opening_tune, 
                        "labels": labels_tune, "filtered_props": filtered_props_tune,
                        "params_used": best_params_tuned # IMPORTANT
                    }
                    if best_diff == 0: break
            if best_diff == 0: break
        
        # st.success(f"Auto-tuning terminé. Meilleurs params: Flou={best_params_tuned['blur_kernel']}, C={best_params_tuned['adapt_c']}")
        return processed_results_for_best_tune # Retourne les résultats avec les params tunés

    # --- Traitement standard (ou après auto-tuning si non retourné plus tôt) ---
    blur_kernel_to_use = current_params_to_use["blur_kernel"]
    adapt_block_size_to_use = current_params_to_use["adapt_block_size"]
    adapt_c_to_use = current_params_to_use["adapt_c"]
    min_area_to_use = current_params_to_use["min_area"]
    morph_kernel_to_use = current_params_to_use["morph_kernel"]
    morph_iterations_to_use = current_params_to_use["morph_iterations"]


    if blur_kernel_to_use > 0:
        blur_k_odd_final = blur_kernel_to_use if blur_kernel_to_use % 2 != 0 else blur_kernel_to_use + 1
        blurred_img_final = cv2.GaussianBlur(gray, (blur_k_odd_final, blur_k_odd_final), 0)
    else:
        blurred_img_final = gray.copy()

    adapt_b_s_odd_final = adapt_block_size_to_use if adapt_block_size_to_use % 2 != 0 else adapt_block_size_to_use + 1
    if adapt_b_s_odd_final <= 1: adapt_b_s_odd_final = 3
    morph_k_odd_final = morph_kernel_to_use if morph_kernel_to_use % 2 != 0 else morph_kernel_to_use + 1
    if morph_k_odd_final < 1: morph_k_odd_final = 1
    
    thresh_final = cv2.adaptiveThreshold(blurred_img_final, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_b_s_odd_final, adapt_c_to_use)
    kernel_closing_final = np.ones((morph_k_odd_final, morph_k_odd_final), np.uint8)
    closing_final = cv2.morphologyEx(thresh_final, cv2.MORPH_CLOSE, kernel_closing_final, iterations=morph_iterations_to_use)
    kernel_opening_final = np.ones((morph_k_odd_final, morph_k_odd_final), np.uint8)
    opening_final = cv2.morphologyEx(closing_final, cv2.MORPH_OPEN, kernel_opening_final, iterations=max(1, morph_iterations_to_use // 2))
    cleared_final = clear_border(opening_final)
    labels_final = measure.label(cleared_final)
    props_final = measure.regionprops(labels_final)
    
    pre_filter_props_final = [p for p in props_final if p.area >= min_area_to_use]

    if use_circularity:
        final_filtered_props_circ = []
        for prop_item in pre_filter_props_final:
            perimeter = prop_item.perimeter
            if perimeter > 0:
                circularity_val = 4 * np.pi * prop_item.area / (perimeter * perimeter)
                if circularity_val >= min_circularity:
                    final_filtered_props_circ.append(prop_item)
        filtered_props_final = final_filtered_props_circ
    else:
        filtered_props_final = pre_filter_props_final

    if apply_relative_filter and len(filtered_props_final) > 1:
        areas = [p.area for p in filtered_props_final]
        if areas: 
            avg_area = np.mean(areas)
            if avg_area > max(1.5 * min_area_to_use, 50):
                relative_threshold_area = 0.1 * avg_area
                final_relative_threshold = max(relative_threshold_area, min_area_to_use)
                filtered_props_after_relative = [p for p in filtered_props_final if p.area >= final_relative_threshold]
                filtered_props_final = filtered_props_after_relative
    
    return {
        "blurred": blurred_img_final, "thresh": thresh_final, "opening": opening_final, 
        "labels": labels_final, "filtered_props": filtered_props_final, 
        "params_used": current_params_to_use 
    }

def extract_insects(image, filtered_props, margin_val):
    extracted_insects = []
    for i, prop in enumerate(filtered_props):
        minr, minc, maxr, maxc = prop.bbox
        minr_marged = max(0, minr - margin_val)
        minc_marged = max(0, minc - margin_val)
        maxr_marged = min(image.shape[0], maxr + margin_val)
        maxc_marged = min(image.shape[1], maxc + margin_val)
        
        insect_roi = image[minr_marged:maxr_marged, minc_marged:maxc_marged].copy()
        roi_height, roi_width = insect_roi.shape[:2]

        if roi_height == 0 or roi_width == 0: continue
            
        # Créer un masque basé sur les coordonnées de prop DANS la ROI extraite avec marge
        mask_from_coords = np.zeros((roi_height, roi_width), dtype=np.uint8)
        for r_orig, c_orig in prop.coords:
            # Coordonnées relatives à la ROI margée
            r_roi = r_orig - minr_marged
            c_roi = c_orig - minc_marged
            if 0 <= r_roi < roi_height and 0 <= c_roi < roi_width:
                mask_from_coords[r_roi, c_roi] = 255
        
        # Utiliser ce masque précis pour l'extraction finale
        # (Les opérations morphologiques suivantes sont pour affiner ce masque précis)
        kernel_close_initial = np.ones((5,5), np.uint8) # Ajustable
        mask_refined = cv2.morphologyEx(mask_from_coords, cv2.MORPH_CLOSE, kernel_close_initial, iterations=2) # Ajustable

        # Optionnel: trouver le plus grand contour dans ce masque affiné pour enlever les petits artefacts
        contours_refined, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask_for_extraction = np.zeros_like(mask_refined)
        if contours_refined:
            largest_contour_refined = max(contours_refined, key=cv2.contourArea)
            cv2.drawContours(final_mask_for_extraction, [largest_contour_refined], -1, 255, thickness=cv2.FILLED)
        else: # Si pas de contour, le masque initial était peut-être trop petit
            final_mask_for_extraction = mask_refined # Ou retourner None/continue

        if np.sum(final_mask_for_extraction) == 0: continue # Si masque vide après affinage

        mask_3ch = cv2.cvtColor(final_mask_for_extraction, cv2.COLOR_GRAY2BGR)
        white_bg = np.ones_like(insect_roi, dtype=np.uint8) * 255
        insect_on_white = np.where(mask_3ch == 255, insect_roi, white_bg)
        
        square_insect = make_square(insect_on_white, fill_color=(255, 255, 255))
        extracted_insects.append({"image": square_insect, "index": i, "original_prop": prop})
    return extracted_insects

@st.cache_resource
def load_saved_model_and_labels(model_dir_path, labels_path_arg):
    model_layer = None
    class_names_loaded = None
    try:
        abs_model_path = os.path.abspath(model_dir_path)
        if not (os.path.exists(abs_model_path) and os.path.isdir(abs_model_path) and os.path.exists(os.path.join(abs_model_path, "saved_model.pb"))):
            print(f"DEBUG: Chemin du modèle invalide ou incomplet: {abs_model_path}")
            return None, None
        model_layer = tf.keras.layers.TFSMLayer(abs_model_path, call_endpoint='serving_default')
        abs_labels_path = os.path.abspath(labels_path_arg)
        if not os.path.exists(abs_labels_path):
            print(f"DEBUG: Fichier de labels introuvable: {abs_labels_path}")
            return model_layer, None
        with open(abs_labels_path, "r") as f:
            class_names_loaded = [line.strip().split(" ", 1)[1] if " " in line.strip() else line.strip() for line in f.readlines()]
        return model_layer, class_names_loaded
    except Exception as e:
        print(f"DEBUG: Erreur chargement modèle/labels: {e}")
        return model_layer, class_names_loaded

def predict_insect_saved_model(image_cv2, model_layer_arg, class_names_arg, input_size):
    if model_layer_arg is None or class_names_arg is None:
        return "Erreur Modèle/Labels", 0.0, []
    # ... (logique de prédiction inchangée)
    img_resized = cv2.resize(image_cv2, input_size, interpolation=cv2.INTER_AREA)
    image_array = np.asarray(img_resized, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1.0
    input_tensor = tf.convert_to_tensor(normalized_image_array)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    predictions_np = None
    try:
        predictions_output = model_layer_arg(input_tensor)
        if isinstance(predictions_output, dict):
            if len(predictions_output) == 1: predictions_tensor = list(predictions_output.values())[0]
            elif 'outputs' in predictions_output: predictions_tensor = predictions_output['outputs']
            elif 'output_0' in predictions_output: predictions_tensor = predictions_output['output_0']
            else:
                key_found = None
                for key, value in predictions_output.items():
                    if isinstance(value, tf.Tensor) and len(value.shape) == 2 and value.shape[0] == 1:
                        predictions_tensor = value; key_found = key; break
                if key_found is None: return "Erreur Sortie Modèle Dict", 0.0, []
        else: predictions_tensor = predictions_output
        if hasattr(predictions_tensor, 'numpy'): predictions_np = predictions_tensor.numpy()
        else: predictions_np = np.array(predictions_tensor)
    except Exception as e_predict: print(f"DEBUG: Erreur prédiction: {e_predict}"); return "Erreur Prédiction", 0.0, []
    if predictions_np is None or predictions_np.size == 0: return "Erreur Prédiction Vide", 0.0, []
    predicted_class_index = np.argmax(predictions_np[0])
    confidence_score = predictions_np[0][predicted_class_index]
    if predicted_class_index >= len(class_names_arg): return "Erreur Index Label", confidence_score, predictions_np[0]
    label_name = class_names_arg[predicted_class_index]
    return label_name, confidence_score, predictions_np[0]


def create_label_display_image(label_image_data, filtered_props):
    # ... (inchangé)
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

def main():
    st.set_page_config(layout="wide")
    st.title("Détection, isolation et identification dʼinsectes")

    # Initialisation de st.session_state
    if 'image_data_list' not in st.session_state:
        st.session_state.image_data_list = []
    if 'model_obj' not in st.session_state: # Renommé pour clarté
        st.session_state.model_obj = None
        st.session_state.class_names_list = None # Renommé

    if st.session_state.model_obj is None:
        model_loaded, class_names_loaded = load_saved_model_and_labels(SAVED_MODEL_DIR_PATH, LABELS_PATH)
        if model_loaded and class_names_loaded:
            st.session_state.model_obj = model_loaded
            st.session_state.class_names_list = class_names_loaded
            st.success("Modèle d'identification et labels chargés.")
        elif model_loaded and not class_names_loaded:
            st.session_state.model_obj = model_loaded
            st.warning("Modèle chargé, mais échec du chargement des labels.")
        else:
            st.error("Échec du chargement du modèle d'identification.")
    
    model_to_use = st.session_state.model_obj
    class_names_to_use = st.session_state.class_names_list

    tab1, tab2, tab3 = st.tabs(["Segmentation et Paramètres", "Identification et Analyse", "Guide dʼutilisation"])

    with tab1:
        st.header("1. Téléversement et Configuration des Images")
        
        uploaded_files = st.file_uploader(
            "Choisissez une ou plusieurs images", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="main_file_uploader_tab1"
        )

        if uploaded_files:
            new_uploaded_file_ids = {f.file_id + "_" + f.name for f in uploaded_files}
            current_img_data_ids = {img_data["id"] for img_data in st.session_state.image_data_list}

            # Ajouter nouvelles images
            for uploaded_file in uploaded_files:
                img_id = uploaded_file.file_id + "_" + uploaded_file.name
                if img_id not in current_img_data_ids:
                    img_bytes = uploaded_file.getvalue()
                    img_cv = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    st.session_state.image_data_list.append({
                        "id": img_id, "filename": uploaded_file.name,
                        "image_bytes": img_bytes, "cv_image": img_cv,
                        "params": DEFAULT_SEG_PARAMS.copy(),
                        "processed_data": None, "is_processed": False, "show_params_expander": False
                    })
            
            # Supprimer anciennes images
            st.session_state.image_data_list = [
                img_data for img_data in st.session_state.image_data_list if img_data["id"] in new_uploaded_file_ids
            ]
        
        if not st.session_state.image_data_list:
            st.info("Veuillez téléverser des images.")
        
        total_detected_overall_seg = 0
        total_expected_overall_seg = 0

        for idx, img_data_item in enumerate(st.session_state.image_data_list):
            st.markdown(f"--- \n ### Image {idx + 1}: {img_data_item['filename']}")
            
            # Utiliser des clés uniques pour les boutons basées sur l'ID de l'image
            show_params_key = f"show_params_btn_{img_data_item['id']}"
            autotune_key = f"autotune_btn_{img_data_item['id']}"
            process_key = f"process_btn_{img_data_item['id']}"

            col_img_disp, col_actions = st.columns([3,1])

            with col_img_disp:
                st.image(cv2.cvtColor(img_data_item["cv_image"], cv2.COLOR_BGR2RGB), caption="Originale", use_column_width=True)
                if img_data_item["is_processed"] and img_data_item["processed_data"]:
                    col_morph_res, col_labels_res = st.columns(2)
                    with col_morph_res:
                        st.image(img_data_item["processed_data"]["opening"], channels="GRAY", caption="Résultat Morphologique", use_column_width=True)
                    with col_labels_res:
                        label_disp_img = create_label_display_image(img_data_item["processed_data"]["labels"], img_data_item["processed_data"]["filtered_props"])
                        st.image(label_disp_img, caption=f"Détectés: {len(img_data_item['processed_data']['filtered_props'])}", use_column_width=True)
            
            with col_actions:
                if st.button(f"Gérer Paramètres", key=show_params_key):
                    img_data_item["show_params_expander"] = not img_data_item["show_params_expander"]
                    # st.rerun() # Optionnel, l'expander devrait se mettre à jour

            if img_data_item["show_params_expander"]:
                with st.expander(f"Paramètres pour {img_data_item['filename']}", expanded=True):
                    params_ref = img_data_item["params"] # Travail direct sur les params de l'image

                    params_ref["target_insect_count"] = st.number_input("Insectes attendus", 0, 100, params_ref["target_insect_count"], 1, key=f"target_{idx}")
                    params_ref["blur_kernel"] = st.slider("Flou (0=aucun)", 0, 21, params_ref["blur_kernel"], 1, key=f"blur_{idx}")
                    params_ref["adapt_block_size"] = st.slider("Bloc Adapt.", 3, 51, params_ref["adapt_block_size"], 2, key=f"block_{idx}")
                    params_ref["adapt_c"] = st.slider("Constante C", -20, 20, params_ref["adapt_c"], 1, key=f"c_{idx}")
                    params_ref["min_area"] = st.slider("Aire Min", 10, 10000, params_ref["min_area"], 10, key=f"area_{idx}")
                    params_ref["morph_kernel"] = st.slider("Noyau Morpho", 1, 15, params_ref["morph_kernel"], 2, key=f"morph_k_{idx}")
                    params_ref["morph_iterations"] = st.slider("It. Morpho", 1, 5, params_ref["morph_iterations"], 1, key=f"morph_i_{idx}")
                    params_ref["margin"] = st.slider("Marge Ext.", 0, 50, params_ref["margin"], key=f"margin_{idx}")
                    params_ref["use_circularity"] = st.checkbox("Filtre Circ.", params_ref["use_circularity"], key=f"circ_c_{idx}")
                    if params_ref["use_circularity"]:
                        params_ref["min_circularity"] = st.slider("Circ. Min Val", 0.0, 1.0, params_ref["min_circularity"], 0.05, key=f"circ_v_{idx}")
                    params_ref["apply_relative_filter"] = st.checkbox("Filtre Relatif", params_ref["apply_relative_filter"], key=f"rel_f_{idx}")

                    if st.button("Auto-ajuster cette image", key=autotune_key):
                        if params_ref["target_insect_count"] > 0:
                            with st.spinner("Auto-ajustement en cours..."):
                                tuned_results = process_image(img_data_item["cv_image"], params_ref.copy(), 
                                                              target_insect_count_for_tune=params_ref["target_insect_count"], 
                                                              auto_tune_mode=True)
                                if tuned_results: # S'assurer que le tuning a retourné quelque chose
                                    img_data_item["params"] = tuned_results["params_used"]
                                    img_data_item["processed_data"] = tuned_results
                                    img_data_item["is_processed"] = True
                                    st.success(f"Auto-ajustement terminé pour {img_data_item['filename']}.")
                                else:
                                    st.error("L'auto-ajustement a échoué.")
                                st.rerun()
                        else:
                            st.warning("Définir 'Insectes attendus' > 0 pour l'auto-ajustement.")
                    
                    if st.button("Traiter cette image", key=process_key):
                        with st.spinner(f"Traitement de {img_data_item['filename']}..."):
                            img_data_item["processed_data"] = process_image(img_data_item["cv_image"], params_ref)
                            img_data_item["is_processed"] = True
                            st.rerun()
            
            if img_data_item["is_processed"] and img_data_item["processed_data"]:
                num_detected = len(img_data_item["processed_data"]["filtered_props"])
                target_count = img_data_item["params"].get("target_insect_count", 0)
                total_detected_overall_seg += num_detected
                total_expected_overall_seg += target_count
                st.metric(label=f"Détectés pour {img_data_item['filename']}", value=num_detected, 
                          delta=f"{num_detected - target_count} vs Attendu" if target_count > 0 else None)

        if st.session_state.image_data_list and any(img['is_processed'] for img in st.session_state.image_data_list):
            st.markdown("--- \n ### Résultats Globaux de Segmentation")
            st.metric("Total insectes détectés (images traitées)", total_detected_overall_seg)
            if total_expected_overall_seg > 0 : # N'afficher que si au moins une cible a été définie
                 st.metric("Total insectes attendus (cibles définies)", total_expected_overall_seg)


    with tab2:
        st.header("Identification et Analyse Écologique")
        if model_to_use is None or class_names_to_use is None:
            st.error("Modèle d'identification ou labels non disponibles.")
        elif not st.session_state.image_data_list or not any(img_data["is_processed"] for img_data in st.session_state.image_data_list):
            st.info("Veuillez d'abord téléverser et traiter des images.")
        else:
            all_identified_labels_for_pie_chart = []
            images_with_processed_insects = [img for img in st.session_state.image_data_list if img["is_processed"] and img["processed_data"]]

            if not images_with_processed_insects:
                 st.info("Aucune image n'a été traitée avec succès pour l'identification.")
            else:
                for img_data_item_id in images_with_processed_insects:
                    extracted_insects_id = extract_insects(
                        img_data_item_id["cv_image"], 
                        img_data_item_id["processed_data"]["filtered_props"], 
                        img_data_item_id["params"]["margin"]
                    )
                    for insect_item in extracted_insects_id:
                        label_id, _, _ = predict_insect_saved_model(
                            insect_item["image"], model_to_use, class_names_to_use, MODEL_INPUT_SIZE
                        )
                        if "Erreur" not in label_id:
                            all_identified_labels_for_pie_chart.append(label_id)
                
                if not all_identified_labels_for_pie_chart:
                    st.warning("Aucun insecte n'a pu être identifié sur l'ensemble des images.")
                else:
                    ecological_counts_pie = {}
                    labels_non_mappes_pie = set()
                    for label_pie in all_identified_labels_for_pie_chart:
                        if label_pie not in ECOLOGICAL_FUNCTIONS_MAP:
                            labels_non_mappes_pie.add(label_pie)
                        eco_func = ECOLOGICAL_FUNCTIONS_MAP.get(label_pie, DEFAULT_ECOLOGICAL_FUNCTION)
                        ecological_counts_pie[eco_func] = ecological_counts_pie.get(eco_func, 0) + 1
                    
                    if labels_non_mappes_pie:
                        st.warning(f"Labels globaux non mappés pour pie chart: {labels_non_mappes_pie}")

                    if ecological_counts_pie:
                        st.subheader("Répartition Globale des Fonctions Écologiques")
                        # ... (code du pie chart et Shannon comme dans la version précédente)
                        labels_pie_keys = list(ecological_counts_pie.keys())
                        sizes_pie_values = list(ecological_counts_pie.values())
                        colors_map_pie = {"Décomposeurs": "#8B4513", "Pollinisateurs": "#FFD700", "Prédateurs": "#DC143C", "Ravageur": "#FF8C00", "Non défini": "#D3D3D3"}
                        pie_colors_list = [colors_map_pie.get(lbl, "#CCCCCC") for lbl in labels_pie_keys]
                        
                        fig_pie, ax_pie = plt.subplots()
                        ax_pie.pie(sizes_pie_values, labels=labels_pie_keys, autopct='%1.1f%%', startangle=90, colors=pie_colors_list)
                        ax_pie.axis('equal')
                        st.pyplot(fig_pie)

                        shannon_idx_val = calculate_shannon_index(ecological_counts_pie)
                        st.subheader("Indice de Shannon Fonctionnel Global (H')")
                        st.metric(label="H'", value=f"{shannon_idx_val:.3f}")
                        if shannon_idx_val == 0 and sum(ecological_counts_pie.values()) > 0:
                            st.caption("Un indice de 0 signifie qu'une seule fonction écologique est présente.")
                        elif shannon_idx_val > 0:
                            max_shannon_val = math.log(len(ecological_counts_pie)) if len(ecological_counts_pie) > 0 else 0
                            st.caption(f"Max H' possible pour {len(ecological_counts_pie)} fonctions: {max_shannon_val:.3f}.")
                    else:
                        st.write("Aucune fonction écologique à afficher pour le pie chart.")
            
            st.markdown("--- \n ### Identification Détaillée par Image")
            for idx, img_data_item_detail in enumerate(images_with_processed_insects):
                st.markdown(f"#### {img_data_item_detail['filename']}")
                # ... (code d'affichage détaillé par image comme avant)
                extracted_insects_detail = extract_insects(
                    img_data_item_detail["cv_image"], 
                    img_data_item_detail["processed_data"]["filtered_props"], 
                    img_data_item_detail["params"]["margin"]
                )
                if not extracted_insects_detail:
                    st.write("Aucun insecte extrait pour identification sur cette image.")
                    continue
                
                num_cols_id_detail = 3
                cols_id_detail = st.columns(num_cols_id_detail)
                col_idx_id_detail = 0
                for insect_detail_item in extracted_insects_detail:
                    label_detail, confidence_detail, _ = predict_insect_saved_model(
                        insect_detail_item["image"], model_to_use, class_names_to_use, MODEL_INPUT_SIZE
                    )
                    with cols_id_detail[col_idx_id_detail % num_cols_id_detail]:
                        st.image(cv2.cvtColor(insect_detail_item["image"], cv2.COLOR_BGR2RGB), caption=f"Insecte #{insect_detail_item['index'] + 1}", width=150)
                        if "Erreur" in label_detail:
                            st.error(f"{label_detail} ({confidence_detail*100:.2f}%)")
                        else:
                            st.markdown(f"**Label:** {label_detail}")
                            st.markdown(f"**Fonction:** {ECOLOGICAL_FUNCTIONS_MAP.get(label_detail, DEFAULT_ECOLOGICAL_FUNCTION)}")
                            st.markdown(f"**Confiance:** {confidence_detail*100:.2f}%")
                    col_idx_id_detail += 1
                st.markdown("---")

    with tab3:
        st.header("Guide dʼutilisation")
        st.subheader("Paramètres de Segmentation (par image)")
        st.write("""
        Pour chaque image téléversée, vous pouvez configurer individuellement les paramètres de segmentation en cliquant sur le bouton "Gérer Paramètres".
        - **Insectes attendus**: Nombre cible pour l'auto-ajustement sur CETTE image.
        - **Auto-ajuster cette image**: Tente d'optimiser 'Flou' et 'Constante C' pour atteindre le nombre d'insectes attendus. Utilise les autres paramètres (Aire Min, Morpho, etc.) que vous avez définis pour cette image.
        - **Traiter cette image**: Applique les paramètres actuels à cette image.
        Les résultats (images originales, morphologiques, labels colorés) sont affichés sous chaque image après traitement.
        """)
        st.subheader("Analyse Écologique (Onglet Identification)")
        st.write("""
        Après avoir traité au moins une image, l'onglet "Identification et Analyse" affiche:
        - Un graphique circulaire (camembert) de la répartition globale des fonctions écologiques.
        - L'Indice de Shannon Fonctionnel global.
        - Une identification détaillée insecte par insecte pour chaque image traitée.
        """)


if __name__ == "__main__":
    if 'image_data_list' not in st.session_state:
        st.session_state.image_data_list = []
    if 'model_obj' not in st.session_state:
        st.session_state.model_obj = None
    if 'class_names_list' not in st.session_state:
        st.session_state.class_names_list = None
    main()

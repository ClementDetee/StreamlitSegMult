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
    "Carabide": "Prédateurs",
    "Arachnides": "Prédateurs",
    "Mouches des semis": "Ravageur"
}
DEFAULT_ECOLOGICAL_FUNCTION = "Non défini"

DEFAULT_SEG_PARAMS = {
    "target_insect_count": 3, "blur_kernel": 5, "adapt_block_size": 35, "adapt_c": 5,
    "min_area": 150, "morph_kernel": 3, "morph_iterations": 2, "margin": 15,
    "use_circularity": False, "min_circularity": 0.3, "apply_relative_filter": True
}

# --- Fonctions Utilitaires ---
# ... (make_square, calculate_shannon_index - inchangées)
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
# ... (process_image, extract_insects, load_saved_model_and_labels, predict_insect_saved_model, create_label_display_image - inchangées par rapport à la version précédente)
# J'omets leur code ici pour la lisibilité, mais elles doivent être présentes.
# Assurez-vous d'avoir les versions de ces fonctions de ma réponse précédente.
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
    
    current_params_to_use = params.copy() 

    if auto_tune_mode and target_insect_count_for_tune > 0:
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

                num_detected_test = len(filtered_props_tune)
                current_diff = abs(num_detected_test - target_insect_count_for_tune)

                if current_diff < best_diff:
                    best_diff = current_diff
                    best_params_tuned = temp_params.copy()
                    processed_results_for_best_tune = {
                        "blurred": blurred_tune, "thresh": thresh_tune, "opening": opening_tune, 
                        "labels": labels_tune, "filtered_props": filtered_props_tune,
                        "params_used": best_params_tuned 
                    }
                    if best_diff == 0: break
            if best_diff == 0: break
        
        return processed_results_for_best_tune 

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
            
        mask_from_coords = np.zeros((roi_height, roi_width), dtype=np.uint8)
        for r_orig, c_orig in prop.coords:
            r_roi = r_orig - minr_marged
            c_roi = c_orig - minc_marged
            if 0 <= r_roi < roi_height and 0 <= c_roi < roi_width:
                mask_from_coords[r_roi, c_roi] = 255
        
        kernel_close_initial = np.ones((5,5), np.uint8)
        mask_refined = cv2.morphologyEx(mask_from_coords, cv2.MORPH_CLOSE, kernel_close_initial, iterations=2)

        contours_refined, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask_for_extraction = np.zeros_like(mask_refined)
        if contours_refined:
            largest_contour_refined = max(contours_refined, key=cv2.contourArea)
            cv2.drawContours(final_mask_for_extraction, [largest_contour_refined], -1, 255, thickness=cv2.FILLED)
        else: 
            final_mask_for_extraction = mask_refined

        if np.sum(final_mask_for_extraction) == 0: continue

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
        return model_layer, class_names_loaded # Retourner ce qui a été chargé

def predict_insect_saved_model(image_cv2, model_layer_arg, class_names_arg, input_size):
    if model_layer_arg is None or class_names_arg is None:
        return "Erreur Modèle/Labels", 0.0, []
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
    if 'image_data_list' not in st.session_state: st.session_state.image_data_list = []
    if 'model_obj' not in st.session_state: st.session_state.model_obj = None
    if 'class_names_list' not in st.session_state: st.session_state.class_names_list = None
    if 'active_image_idx_for_params' not in st.session_state: st.session_state.active_image_idx_for_params = None # Pour la sidebar

    if st.session_state.model_obj is None: # Charger le modèle une fois
        model_loaded, class_names_loaded = load_saved_model_and_labels(SAVED_MODEL_DIR_PATH, LABELS_PATH)
        if model_loaded:
            st.session_state.model_obj = model_loaded
            if class_names_loaded:
                st.session_state.class_names_list = class_names_loaded
                # st.success("Modèle et labels chargés.") # Peut être affiché une fois
            else: st.warning("Modèle chargé, mais échec du chargement des labels.")
        else: st.error("Échec du chargement du modèle d'identification.")
    
    model_to_use = st.session_state.model_obj
    class_names_to_use = st.session_state.class_names_list

    # --- Sidebar pour les paramètres de l'image active ---
    with st.sidebar:
        st.header("Paramètres de Segmentation")
        active_idx = st.session_state.active_image_idx_for_params
        
        if active_idx is not None and active_idx < len(st.session_state.image_data_list):
            st.markdown(f"**Pour : {st.session_state.image_data_list[active_idx]['filename']}**")
            active_img_params = st.session_state.image_data_list[active_idx]["params"]

            active_img_params["target_insect_count"] = st.number_input(
                "Insectes attendus", 0, 100, active_img_params["target_insect_count"], 1, 
                key=f"sidebar_target_{active_idx}"
            )
            active_img_params["blur_kernel"] = st.slider(
                "Flou (0=aucun)", 0, 21, active_img_params["blur_kernel"], 1, 
                key=f"sidebar_blur_{active_idx}"
            )
            active_img_params["adapt_block_size"] = st.slider(
                "Bloc Adapt.", 3, 51, active_img_params["adapt_block_size"], 2, 
                key=f"sidebar_block_{active_idx}"
            )
            active_img_params["adapt_c"] = st.slider(
                "Constante C", -20, 20, active_img_params["adapt_c"], 1, 
                key=f"sidebar_c_{active_idx}"
            )
            active_img_params["min_area"] = st.slider(
                "Aire Min", 10, 10000, active_img_params["min_area"], 10, 
                key=f"sidebar_area_{active_idx}"
            )
            active_img_params["morph_kernel"] = st.slider(
                "Noyau Morpho", 1, 15, active_img_params["morph_kernel"], 2, 
                key=f"sidebar_morph_k_{active_idx}"
            )
            active_img_params["morph_iterations"] = st.slider(
                "It. Morpho", 1, 5, active_img_params["morph_iterations"], 1, 
                key=f"sidebar_morph_i_{active_idx}"
            )
            active_img_params["margin"] = st.slider("Marge Ext.", 0, 50, active_img_params["margin"], key=f"sidebar_margin_{active_idx}")
            active_img_params["use_circularity"] = st.checkbox("Filtre Circ.", active_img_params["use_circularity"], key=f"sidebar_circ_c_{active_idx}")
            if active_img_params["use_circularity"]:
                active_img_params["min_circularity"] = st.slider("Circ. Min Val", 0.0, 1.0, active_img_params["min_circularity"], 0.05, key=f"sidebar_circ_v_{active_idx}")
            active_img_params["apply_relative_filter"] = st.checkbox("Filtre Relatif", active_img_params["apply_relative_filter"], key=f"sidebar_rel_f_{active_idx}")

            if st.button("Appliquer et Traiter cette image", key=f"sidebar_apply_btn_{active_idx}"):
                with st.spinner(f"Traitement de {st.session_state.image_data_list[active_idx]['filename']}..."):
                    st.session_state.image_data_list[active_idx]["processed_data"] = process_image(
                        st.session_state.image_data_list[active_idx]["cv_image"], 
                        active_img_params # Utilise les params modifiés de la sidebar
                    )
                    st.session_state.image_data_list[active_idx]["is_processed"] = True
                    st.rerun() # Pour rafraîchir l'affichage dans l'onglet principal

            if st.button("Auto-ajuster cette image", key=f"sidebar_autotune_btn_{active_idx}"):
                if active_img_params["target_insect_count"] > 0:
                    with st.spinner("Auto-ajustement en cours..."):
                        tuned_results = process_image(
                            st.session_state.image_data_list[active_idx]["cv_image"], 
                            active_img_params.copy(), # Passer une copie pour que le tuning ne modifie pas direct
                            target_insect_count_for_tune=active_img_params["target_insect_count"], 
                            auto_tune_mode=True
                        )
                        if tuned_results:
                            st.session_state.image_data_list[active_idx]["params"] = tuned_results["params_used"]
                            st.session_state.image_data_list[active_idx]["processed_data"] = tuned_results
                            st.session_state.image_data_list[active_idx]["is_processed"] = True
                            st.success(f"Auto-ajustement terminé. Nouveaux params appliqués pour {st.session_state.image_data_list[active_idx]['filename']}.")
                        else:
                            st.error("L'auto-ajustement a échoué à trouver une solution.")
                        st.rerun()
                else:
                    st.warning("Définir 'Insectes attendus' > 0 pour l'auto-ajustement.")
        else:
            st.info("Sélectionnez une image dans l'onglet 'Segmentation' pour configurer ses paramètres ici.")


    tab1, tab2, tab3 = st.tabs(["Segmentation", "Identification et Analyse", "Guide dʼutilisation"])

    with tab1:
        st.header("Gestion des Images et Segmentation")
        
        uploaded_files = st.file_uploader(
            "1. Choisissez vos images", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="tab1_file_uploader"
        )

        if uploaded_files:
            new_uploaded_file_ids = {f.file_id + "_" + f.name for f in uploaded_files}
            current_img_data_ids_in_session = {img_d["id"] for img_d in st.session_state.image_data_list}
            for uploaded_file_item in uploaded_files:
                img_id_new = uploaded_file_item.file_id + "_" + uploaded_file_item.name
                if img_id_new not in current_img_data_ids_in_session:
                    img_bytes_new = uploaded_file_item.getvalue()
                    img_cv_new = cv2.imdecode(np.frombuffer(img_bytes_new, np.uint8), cv2.IMREAD_COLOR)
                    st.session_state.image_data_list.append({
                        "id": img_id_new, "filename": uploaded_file_item.name,
                        "image_bytes": img_bytes_new, "cv_image": img_cv_new,
                        "params": DEFAULT_SEG_PARAMS.copy(),
                        "processed_data": None, "is_processed": False
                    })
            st.session_state.image_data_list = [
                img_d for img_d in st.session_state.image_data_list if img_d["id"] in new_uploaded_file_ids
            ]
            # Si aucune image n'est active et qu'il y a des images, activer la première
            if st.session_state.active_image_idx_for_params is None and st.session_state.image_data_list:
                 st.session_state.active_image_idx_for_params = 0
                 st.rerun() # Pour que la sidebar se mette à jour avec la première image

        if not st.session_state.image_data_list:
            st.info("Veuillez téléverser des images pour commencer.")
        
        # Bouton pour traiter toutes les images avec leurs paramètres individuels
        if st.session_state.image_data_list:
             st.markdown("---")
             if st.button("Traiter TOUTES les images (avec leurs paramètres respectifs)", key="process_all_images_btn"):
                 with st.spinner("Traitement de toutes les images en cours..."):
                     for img_idx_all in range(len(st.session_state.image_data_list)):
                         img_data_all = st.session_state.image_data_list[img_idx_all]
                         # st.write(f"Traitement de {img_data_all['filename']}...") # Feedback
                         img_data_all["processed_data"] = process_image(
                             img_data_all["cv_image"], 
                             img_data_all["params"] # Utilise les params spécifiques de cette image
                         )
                         img_data_all["is_processed"] = True
                 st.success("Toutes les images ont été traitées.")
                 st.rerun()


        for idx, img_data_display in enumerate(st.session_state.image_data_list):
            st.markdown(f"--- \n ### Image {idx + 1}: {img_data_display['filename']}")
            
            cols_display_img = st.columns(3)
            with cols_display_img[0]:
                st.image(cv2.cvtColor(img_data_display["cv_image"], cv2.COLOR_BGR2RGB), caption="Originale", use_column_width=True)
            
            if img_data_display["is_processed"] and img_data_display["processed_data"]:
                with cols_display_img[1]:
                    st.image(img_data_display["processed_data"]["opening"], channels="GRAY", caption="Résultat Morphologique", use_column_width=True)
                with cols_display_img[2]:
                    label_disp_img_main = create_label_display_image(img_data_display["processed_data"]["labels"], img_data_display["processed_data"]["filtered_props"])
                    st.image(label_disp_img_main, caption=f"Détectés: {len(img_data_display['processed_data']['filtered_props'])}", use_column_width=True)
                
                num_detected_disp = len(img_data_display["processed_data"]["filtered_props"])
                target_count_disp = img_data_display["params"].get("target_insect_count", 0)
                st.metric(label=f"Détectés", value=num_detected_disp, 
                          delta=f"{num_detected_disp - target_count_disp} vs Attendu ({target_count_disp})" if target_count_disp > 0 else None)
            else:
                with cols_display_img[1]:
                    st.info("Non traitée ou en attente de traitement.")

            # Bouton pour sélectionner cette image pour édition dans la sidebar
            if st.button(f"Éditer Paramètres pour {img_data_display['filename']}", key=f"select_edit_btn_{img_data_display['id']}"):
                st.session_state.active_image_idx_for_params = idx
                st.rerun() # Pour rafraîchir la sidebar

    with tab2:
        st.header("Identification et Analyse Écologique")
        # ... (Code de l'onglet Identification - la logique de collecte des données et d'affichage du Shannon/pie chart reste la même)
        # Assurez-vous que la taille du pie chart est ajustée.
        if model_to_use is None or class_names_to_use is None:
            st.error("Modèle d'identification ou labels non disponibles.")
        elif not st.session_state.image_data_list or not any(img_d["is_processed"] for img_d in st.session_state.image_data_list):
            st.info("Veuillez d'abord téléverser et traiter des images.")
        else:
            all_identified_labels_pie = []
            images_processed_for_id = [img for img in st.session_state.image_data_list if img["is_processed"] and img["processed_data"]]

            if not images_processed_for_id:
                 st.info("Aucune image traitée disponible pour l'identification.")
            else:
                for img_data_id_item in images_processed_for_id:
                    extracted_insects_id_list = extract_insects(
                        img_data_id_item["cv_image"], 
                        img_data_id_item["processed_data"]["filtered_props"], 
                        img_data_id_item["params"]["margin"]
                    )
                    for insect_id_item in extracted_insects_id_list:
                        label_id_val, _, _ = predict_insect_saved_model(
                            insect_id_item["image"], model_to_use, class_names_to_use, MODEL_INPUT_SIZE
                        )
                        if "Erreur" not in label_id_val:
                            all_identified_labels_pie.append(label_id_val)
                
                if not all_identified_labels_pie:
                    st.warning("Aucun insecte n'a pu être identifié sur l'ensemble des images.")
                else:
                    ecological_counts_for_pie = {}
                    # ... (logique de comptage et de labels non mappés comme avant)
                    labels_non_mappes_pie_glob = set()
                    for label_pie_item in all_identified_labels_pie:
                        if label_pie_item not in ECOLOGICAL_FUNCTIONS_MAP:
                            labels_non_mappes_pie_glob.add(label_pie_item)
                        eco_func_item = ECOLOGICAL_FUNCTIONS_MAP.get(label_pie_item, DEFAULT_ECOLOGICAL_FUNCTION)
                        ecological_counts_for_pie[eco_func_item] = ecological_counts_for_pie.get(eco_func_item, 0) + 1
                    
                    if labels_non_mappes_pie_glob:
                        st.warning(f"Labels globaux non mappés pour pie chart: {labels_non_mappes_pie_glob}")


                    if ecological_counts_for_pie:
                        st.subheader("Répartition Globale des Fonctions Écologiques")
                        
                        labels_pie_chart_keys = list(ecological_counts_for_pie.keys())
                        sizes_pie_chart_values = list(ecological_counts_for_pie.values())
                        colors_map_for_pie = {"Décomposeurs": "#8B4513", "Pollinisateurs": "#FFD700", "Prédateurs": "#DC143C", "Ravageur": "#FF8C00", "Non défini": "#D3D3D3"}
                        pie_colors_for_chart = [colors_map_for_pie.get(lbl_p, "#CCCCCC") for lbl_p in labels_pie_chart_keys]
                        
                        # MODIFICATION : Réduire la taille de la figure
                        fig_pie_chart, ax_pie_chart = plt.subplots(figsize=(6, 4)) # Ajustez (width, height) en inches
                        ax_pie_chart.pie(sizes_pie_chart_values, labels=labels_pie_chart_keys, autopct='%1.1f%%', startangle=90, colors=pie_colors_for_chart, textprops={'fontsize': 8}) # Réduire taille police
                        ax_pie_chart.axis('equal')
                        st.pyplot(fig_pie_chart)

                        # Affichage de l'indice de Shannon
                        shannon_functional_idx = calculate_shannon_index(ecological_counts_for_pie)
                        st.subheader("Indice de Shannon Fonctionnel Global (H')")
                        st.metric(label="H'", value=f"{shannon_functional_idx:.3f}")
                        if shannon_functional_idx == 0 and sum(ecological_counts_for_pie.values()) > 0:
                            st.caption("Un indice de 0 signifie qu'une seule fonction écologique est présente globalement.")
                        elif shannon_functional_idx > 0:
                            max_shannon_val_info = math.log(len(ecological_counts_for_pie)) if len(ecological_counts_for_pie) > 0 else 0
                            st.caption(f"Max H' possible pour {len(ecological_counts_for_pie)} fonctions: {max_shannon_val_info:.3f}.")
                    else:
                        st.write("Aucune fonction écologique à afficher.")
            
            st.markdown("--- \n ### Identification Détaillée par Image")
            # ... (Affichage détaillé par image comme avant)
            for idx_detail, img_data_item_detail_id in enumerate(images_processed_for_id): # Utiliser la liste filtrée
                st.markdown(f"#### {img_data_item_detail_id['filename']}")
                extracted_insects_detail_id = extract_insects(
                    img_data_item_detail_id["cv_image"], 
                    img_data_item_detail_id["processed_data"]["filtered_props"], 
                    img_data_item_detail_id["params"]["margin"]
                )
                if not extracted_insects_detail_id:
                    st.write("Aucun insecte extrait pour identification sur cette image.")
                    continue
                
                num_cols_id_detail_disp = 3
                cols_id_detail_disp = st.columns(num_cols_id_detail_disp)
                col_idx_id_detail_disp = 0
                for insect_detail_item_id in extracted_insects_detail_id:
                    label_detail_id, confidence_detail_id, _ = predict_insect_saved_model(
                        insect_detail_item_id["image"], model_to_use, class_names_to_use, MODEL_INPUT_SIZE
                    )
                    with cols_id_detail_disp[col_idx_id_detail_disp % num_cols_id_detail_disp]:
                        st.image(cv2.cvtColor(insect_detail_item_id["image"], cv2.COLOR_BGR2RGB), caption=f"Insecte #{insect_detail_item_id['index'] + 1}", width=150)
                        if "Erreur" in label_detail_id:
                            st.error(f"{label_detail_id} ({confidence_detail_id*100:.2f}%)")
                        else:
                            st.markdown(f"**Label:** {label_detail_id}")
                            st.markdown(f"**Fonction:** {ECOLOGICAL_FUNCTIONS_MAP.get(label_detail_id, DEFAULT_ECOLOGICAL_FUNCTION)}")
                            st.markdown(f"**Confiance:** {confidence_detail_id*100:.2f}%")
                    col_idx_id_detail_disp += 1
                st.markdown("---")


    with tab3:
        st.header("Guide dʼutilisation")
        st.subheader("Paramètres de Segmentation (par image)")
        st.write("""
        Dans l'onglet "Segmentation et Paramètres":
        1. **Téléversez vos images.**
        2. Pour chaque image, un bouton "Éditer Paramètres pour [nom_image]" apparaît. Cliquez dessus pour afficher ses paramètres dans la **sidebar de gauche**.
        3. Ajustez les paramètres dans la sidebar pour l'image sélectionnée:
            - **Insectes attendus**: Nombre cible pour l'auto-ajustement.
            - **Sliders (Flou, Aire Min, etc.)**: Modifiez les valeurs.
            - **Appliquer et Traiter cette image**: Applique les paramètres de la sidebar à l'image sélectionnée et affiche les résultats de segmentation.
            - **Auto-ajuster cette image**: Tente d'optimiser 'Flou' et 'Constante C' pour l'image sélectionnée.
        4. Un bouton "Traiter TOUTES les images" est disponible pour appliquer les paramètres configurés individuellement à chaque image respective.
        """)
        st.subheader("Analyse Écologique (Onglet Identification)")
        st.write("""
        Après avoir traité au moins une image, cet onglet affiche:
        - Un graphique circulaire (camembert) de la répartition globale des fonctions écologiques (taille ajustée).
        - L'Indice de Shannon Fonctionnel global, qui mesure la diversité de ces fonctions.
        - Une identification détaillée insecte par insecte pour chaque image traitée.
        """)


if __name__ == "__main__":
    if 'image_data_list' not in st.session_state: st.session_state.image_data_list = []
    if 'model_obj' not in st.session_state: st.session_state.model_obj = None
    if 'class_names_list' not in st.session_state: st.session_state.class_names_list = None
    if 'active_image_idx_for_params' not in st.session_state: st.session_state.active_image_idx_for_params = None
    main()

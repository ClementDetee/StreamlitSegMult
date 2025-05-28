    *   Ajouter "Les isopodes peuvent aussi consommer de jeunes pousses (rare et faiblepython
import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage. impact)."

3.  **Taille et position du diagramme + Tableau récapitulatif :**
    *segmentation import clear_border
import os
import io
from PIL import Image, ImageOps
import tempfile
import zip   Diagramme (pie chart) encore plus petit et à droite.
    *   À gauche du diagramme,file
import base64
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import pandas un tableau récapitulatif :
        *   Groupe taxonomique (label prédit)
        *   Quantité as pd # Ajout pour le tableau récapitulatif

# --- Configuration Globale ---
SAVED_MODEL_DIR_PATH = "model.savedmodel"
LABELS_PATH = "labels.txt"
MODEL_INPUT_SIZE = (224, 224)

# MODIFICATION: Label et Map
ECOLOGICAL_
        *   Fonction écologique associée
    *   Sous ce tableau (ou à la fin de cette section), la valeur de l'Indice de Shannon.

**Code `app.py` avec les nouvelles modifications :**

FUNCTIONS_MAP = {
    "Apidae": "Pollinisateurs",
    "Isopodes":Je vais me concentrer sur les changements dans `ECOLOGICAL_FUNCTIONS_MAP`, l'onglet `tab2` pour "Décomposeurs et ingénieurs du sol", # MODIFIÉ ICI
    "Carabide": "Prédateurs",
    "Opiliones et Araneae": "Prédateurs",
    "Mou le tableau et le graphique, et le texte en bas.

```python
import streamlit as st
import cv2ches des semis": "Ravageur"
}
DEFAULT_ECOLOGICAL_FUNCTION = "Non défini"
import numpy as np
from skimage import measure
from skimage.segmentation import clear_border
import os
import io
from PIL import Image, ImageOps
import tempfile
import zipfile
import base64


DEFAULT_SEG_PARAMS = {
    "target_insect_count": 1,
    "blur_kernel": 5, "adapt_block_size": 35, "adapt_c": 5,import tensorflow as tf
import matplotlib.pyplot as plt
import math
import pandas as pd # Ajout pour créer "min_area": 150,
    "morph_kernel": 3, "morph_iterations": 2 facilement le tableau

# --- Configuration Globale ---
SAVED_MODEL_DIR_PATH = "model.savedmodel, "margin": 15, "use_circularity": False,
    "min_circularity": 0."
LABELS_PATH = "labels.txt"
MODEL_INPUT_SIZE = (224,3, "apply_relative_filter": True
}

# --- Fonctions Utilitaires ---
# ... (make_ 224)

# MODIFICATION: Label et Map
ECOLOGICAL_FUNCTIONS_MAP = {
    "Apidae": "Pollinisateurs",
    "Isopodes": "Décomposeurs et Ingénieurssquare, calculate_shannon_index - inchangées)
def make_square(image, fill_color du sol", # MODIFIÉ ICI
    "Carabide": "Prédateurs",
    "Op=(255, 255, 255)):
    height, width = image.shape[:2]
    max_side = max(height, width)
    top = (max_side -iliones et Araneae": "Prédateurs",
    "Mouches des semis": "Ravageur height) // 2
    bottom = max_side - height - top
    left = (max_side"
}
DEFAULT_ECOLOGICAL_FUNCTION = "Non défini"

DEFAULT_SEG_PARAMS = {
     - width) // 2
    right = max_side - width - left
    square_image = cv"target_insect_count": 1, "blur_kernel": 5, "adapt_block_size2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value": 35, "adapt_c": 5,
    "min_area": 150=fill_color)
    return square_image

def calculate_shannon_index(counts_dict):, "morph_kernel": 3, "morph_iterations": 2, "margin": 15,
    "
    if not counts_dict or sum(counts_dict.values()) == 0:
        return use_circularity": False, "min_circularity": 0.3, "apply_relative_filter0.0
    total_individuals = sum(counts_dict.values())
    shannon_index =": True
}

# --- Fonctions Utilitaires ---
# ... (make_square, calculate_sh 0.0
    for category_count in counts_dict.values():
        if category_count >annon_index - inchangées)
def make_square(image, fill_color=(255, 0:
            proportion = category_count / total_individuals
            shannon_index -= proportion * math.log(proportion)
    return shannon_index

# --- Fonctions de Traitement d'Image et 255, 255)):
    height, width = image.shape[:2]
    max_side = max(height, width)
    top = (max_side - height) // 2 Modèle ---
# ... (process_image, extract_insects, load_saved_model_and_
    bottom = max_side - height - top
    left = (max_side - width) // labels, predict_insect_saved_model, create_label_display_image)
# Ces fonctions restent les mêmes que2
    right = max_side - width - left
    square_image = cv2.copyMakeBorder dans la version précédente. Je les inclus pour la complétude.
def process_image(image_cv, params):(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    return square_image

def calculate_shannon_index(counts_dict):
    if not counts
    blur_kernel = params["blur_kernel"]
    adapt_block_size = params["adapt_block_size"]
    adapt_c = params["adapt_c"]
    min_area_param =_dict or sum(counts_dict.values()) == 0:
        return 0.0
     params["min_area"]
    morph_kernel_size = params["morph_kernel"]
    morph_total_individuals = sum(counts_dict.values())
    shannon_index = 0.0
    for category_count in counts_dict.values():
        if category_count > 0:
            iterations = params["morph_iterations"]
    use_circularity = params.get("use_circularity", False)
    min_circularity = params.get("min_circularity", 0.3)
proportion = category_count / total_individuals
            shannon_index -= proportion * math.log(proportion)    apply_relative_filter = params.get("apply_relative_filter", True)
    gray = cv
    return shannon_index

# --- Fonctions de Traitement d'Image et Modèle ---
# ... (process_image, extract_insects, load_saved_model_and_labels, predict_insect2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    if blur_kernel > 0:
        blur_k_odd = blur_kernel if blur_kernel % 2 != 0_saved_model, create_label_display_image)
# Ces fonctions restent les mêmes que dans la version précédente. else blur_kernel + 1
        blurred_img = cv2.GaussianBlur(gray, (blur_k_odd,
# Assurez-vous d'avoir les versions de ces fonctions de ma réponse précédente.
def process_image(image_ blur_k_odd), 0)
    else:
        blurred_img = gray.copy()
cv, params):
    blur_kernel = params["blur_kernel"]
    adapt_block_size =    adapt_b_s_odd = adapt_block_size if adapt_block_size % 2 != 0 params["adapt_block_size"]
    adapt_c = params["adapt_c"]
    min_area_param = params["min_area"]
    morph_kernel_size = params["morph_kernel"] else adapt_block_size + 1
    if adapt_b_s_odd <= 1: adapt
    morph_iterations = params["morph_iterations"]
    use_circularity = params.get("use_b_s_odd = 3
    morph_k_odd = morph_kernel_size if morph_circularity", False)
    min_circularity = params.get("min_circularity", 0_kernel_size % 2 != 0 else morph_kernel_size + 1
    if morph_.3)
    apply_relative_filter = params.get("apply_relative_filter", True)

k_odd < 1: morph_k_odd = 1
    thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    if blur_kernel > 0:
        blur_k_odd = blur_kernel if blur_kernel.THRESH_BINARY_INV, adapt_b_s_odd, adapt_c)
    kernel_closing % 2 != 0 else blur_kernel + 1
        blurred_img = cv2.GaussianBlur( = np.ones((morph_k_odd, morph_k_odd), np.uint8)
    gray, (blur_k_odd, blur_k_odd), 0)
    else:
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_closing, iterations=morph_iterationsblurred_img = gray.copy()

    adapt_b_s_odd = adapt_block_size if adapt_block)
    kernel_opening = np.ones((morph_k_odd, morph_k_odd), np_size % 2 != 0 else adapt_block_size + 1
    if adapt_b_.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,s_odd <= 1: adapt_b_s_odd = 3
    morph_k_odd kernel_opening, iterations=max(1, morph_iterations // 2))
    cleared = clear_border(opening)
    labels = measure.label(cleared)
    props = measure.regionprops( = morph_kernel_size if morph_kernel_size % 2 != 0 else morph_kernel_size + 1
    if morph_k_odd < 1: morph_k_odd = 1
labels)
    pre_filter_props = [p for p in props if p.area >= min_area_param    
    thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.AD]
    if use_circularity:
        final_filtered_props_circ = []
        for propAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_b__item in pre_filter_props:
            perimeter = prop_item.perimeter
            if perimeter > s_odd, adapt_c)
    kernel_closing = np.ones((morph_k_odd, morph0:
                circularity_val = 4 * np.pi * prop_item.area / (perimeter * perimeter)
                if circularity_val >= min_circularity:
                    final_filtered_props__k_odd), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPHcirc.append(prop_item)
        filtered_props = final_filtered_props_circ
    else_CLOSE, kernel_closing, iterations=morph_iterations)
    kernel_opening = np.ones((morph:
        filtered_props = pre_filter_props
    if apply_relative_filter and len(filtered_k_odd, morph_k_odd), np.uint8)
    opening = cv2.morph_props) > 1:
        areas = [p.area for p in filtered_props]
        ologyEx(closing, cv2.MORPH_OPEN, kernel_opening, iterations=max(1, morph_if areas: 
            avg_area = np.mean(areas)
            if avg_area > maxiterations // 2))
    cleared = clear_border(opening)
    labels = measure.label(cleared)
    props = measure.regionprops(labels)
    
    pre_filter_props =(1.5 * min_area_param, 50):
                relative_threshold_area = 0.1 * avg_area
                final_relative_threshold = max(relative_threshold_area, min [p for p in props if p.area >= min_area_param]

    if use_circularity:
        final_filtered_props_circ = []
        for prop_item in pre_filter_props_area_param)
                filtered_props_after_relative = [p for p in filtered_props if:
            perimeter = prop_item.perimeter
            if perimeter > 0:
                circularity_val p.area >= final_relative_threshold]
                filtered_props = filtered_props_after_relative
 = 4 * np.pi * prop_item.area / (perimeter * perimeter)
                if circularity    return {"blurred": blurred_img, "thresh": thresh, "opening": opening, 
            "labels": labels_val >= min_circularity:
                    final_filtered_props_circ.append(prop_item), "filtered_props": filtered_props, 
            "params_used": params.copy()}

def extract_in
        filtered_props = final_filtered_props_circ
    else:
        filtered_props = pre_filter_sects(image, filtered_props, margin_val):
    extracted_arthropods = []
    forprops

    if apply_relative_filter and len(filtered_props) > 1:
        areas = i, prop in enumerate(filtered_props):
        minr, minc, maxr, maxc = [p.area for p in filtered_props]
        if areas: 
            avg_area = np prop.bbox
        minr_marged = max(0, minr - margin_val); minc_marg.mean(areas)
            if avg_area > max(1.5 * min_area_param,ed = max(0, minc - margin_val)
        maxr_marged = min(image 50):
                relative_threshold_area = 0.1 * avg_area
                final_relative_threshold.shape[0], maxr + margin_val); maxc_marged = min(image.shape[1], = max(relative_threshold_area, min_area_param)
                filtered_props_after_relative = [p for p in filtered_props if p.area >= final_relative_threshold]
                filtered_ maxc + margin_val)
        arthropod_roi = image[minr_marged:maxr_marged, minc_marged:maxc_marged].copy()
        roi_height, roiprops = filtered_props_after_relative
    
    return {
        "blurred": blurred_img,_width = arthropod_roi.shape[:2]
        if roi_height == 0 or roi_ "thresh": thresh, "opening": opening, 
        "labels": labels, "filtered_props": filteredwidth == 0: continue
        mask_from_coords = np.zeros((roi_height, roi__props, 
        "params_used": params.copy() 
    }

def extract_inwidth), dtype=np.uint8)
        for r_orig, c_orig in prop.coords:
            r_roi = r_orig - minr_marged; c_roi = c_orig -sects(image, filtered_props, margin_val): # Renommer en extract_arthropods si on est strict
    extracted_arthropods = [] 
    for i, prop in enumerate(filtered_props minc_marged
            if 0 <= r_roi < roi_height and 0 <= c_):
        minr, minc, maxr, maxc = prop.bbox
        minr_margroi < roi_width: mask_from_coords[r_roi, c_roi] = 25ed = max(0, minr - margin_val)
        minc_marged = max(05
        kernel_close_initial = np.ones((5,5), np.uint8) 
        mask, minc - margin_val)
        maxr_marged = min(image.shape[0],_refined = cv2.morphologyEx(mask_from_coords, cv2.MORPH_CLOSE, kernel_close_ maxr + margin_val)
        maxc_marged = min(image.shape[1], maxinitial, iterations=2) 
        contours_refined, _ = cv2.findContours(mask_refinedc + margin_val)
        
        arthropod_roi = image[minr_marged:max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_maskr_marged, minc_marged:maxc_marged].copy() 
        roi_height, roi_for_extraction = np.zeros_like(mask_refined)
        if contours_refined:
            largest_contour_width = arthropod_roi.shape[:2]

        if roi_height == 0 or roi__refined = max(contours_refined, key=cv2.contourArea)
            cv2.drawContourswidth == 0: continue
            
        mask_from_coords = np.zeros((roi_height,(final_mask_for_extraction, [largest_contour_refined], -1, 255, roi_width), dtype=np.uint8)
        for r_orig, c_orig in prop. thickness=cv2.FILLED)
        else: final_mask_for_extraction = mask_refined coords:
            r_roi = r_orig - minr_marged
            c_roi = c
        if np.sum(final_mask_for_extraction) == 0: continue
        mask__orig - minc_marged
            if 0 <= r_roi < roi_height and 03ch = cv2.cvtColor(final_mask_for_extraction, cv2.COLOR_GRAY2BGR <= c_roi < roi_width:
                mask_from_coords[r_roi, c_roi)
        white_bg = np.ones_like(arthropod_roi, dtype=np.uint8)] = 255
        
        kernel_close_initial = np.ones((5,5), * 255
        arthropod_on_white = np.where(mask_3ch == np.uint8) 
        mask_refined = cv2.morphologyEx(mask_from_coords 255, arthropod_roi, white_bg)
        square_arthropod = make_square(, cv2.MORPH_CLOSE, kernel_close_initial, iterations=2) 

        contours_refined, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_arthropod_on_white, fill_color=(255, 255, 255))
        extracted_arthropods.append({"image": square_arthropod, "index": iAPPROX_SIMPLE)
        final_mask_for_extraction = np.zeros_like(mask_refined, "original_prop": prop})
    return extracted_arthropods

@st.cache_resource
def load)
        if contours_refined:
            largest_contour_refined = max(contours_refined, key=cv2.contourArea)
            cv2.drawContours(final_mask_for_extraction, [largest_saved_model_and_labels(model_dir_path, labels_path_arg):
    model_layer = None; class_names_loaded = None
    try:
        abs_model_path =_contour_refined], -1, 255, thickness=cv2.FILLED)
        else os.path.abspath(model_dir_path)
        if not (os.path.exists(abs: 
            final_mask_for_extraction = mask_refined 

        if np.sum(final_model_path) and os.path.isdir(abs_model_path) and os.path.exists_mask_for_extraction) == 0: continue

        mask_3ch = cv2.cvtColor((os.path.join(abs_model_path, "saved_model.pb"))):
            printfinal_mask_for_extraction, cv2.COLOR_GRAY2BGR)
        white_bg = np.ones_like(arthropod_roi, dtype=np.uint8) * 255 
        (f"DEBUG: Modèle invalide: {abs_model_path}"); return None, None
        modelarthropod_on_white = np.where(mask_3ch == 255, arthropod_roi,_layer = tf.keras.layers.TFSMLayer(abs_model_path, call_endpoint='serving_default')
        abs_labels_path = os.path.abspath(labels_path_arg) white_bg) 
        
        square_arthropod = make_square(arthropod_on
        if not os.path.exists(abs_labels_path):
            print(f"DEBUG: Labels_white, fill_color=(255, 255, 255)) 
        extracted_arthropods.append({"image": square_arthropod, "index": i, "original_ introuvables: {abs_labels_path}"); return model_layer, None
        with open(abs_labelsprop": prop}) 
    return extracted_arthropods 

@st.cache_resource
def load_path, "r") as f:
            class_names_raw = [line.strip() for line in f_saved_model_and_labels(model_dir_path, labels_path_arg):
    model.readlines()]
            class_names_loaded = []
            for line in class_names_raw:
                parts =_layer = None
    class_names_loaded = None
    try:
        abs_model_path line.split(" ", 1)
                if len(parts) > 1 and parts[0].isdigit = os.path.abspath(model_dir_path)
        if not (os.path.exists((): class_names_loaded.append(parts[1])
                else: class_names_loaded.append(lineabs_model_path) and os.path.isdir(abs_model_path) and os.path.)
        return model_layer, class_names_loaded
    except Exception as e: print(f"DEBUGexists(os.path.join(abs_model_path, "saved_model.pb"))):
            print(f"DEBUG: Chemin du modèle invalide ou incomplet: {abs_model_path}")
            : Erreur chargement modèle/labels: {e}"); return model_layer, class_names_loaded

def predict_insect_saved_model(image_cv2, model_layer_arg, class_names_argreturn None, None
        model_layer = tf.keras.layers.TFSMLayer(abs_model_path,, input_size):
    if model_layer_arg is None or class_names_arg is None: call_endpoint='serving_default')
        abs_labels_path = os.path.abspath(labels_path_arg return "Erreur Modèle/Labels", 0.0, []
    img_resized = cv2.)
        if not os.path.exists(abs_labels_path):
            print(f"DEBUGresize(image_cv2, input_size, interpolation=cv2.INTER_AREA)
    image_array = np: Fichier de labels introuvable: {abs_labels_path}")
            return model_layer, None
        with open(abs_labels_path, "r") as f:
            class_names_raw.asarray(img_resized, dtype=np.float32); normalized_image_array = (image_array / 127.5) - 1.0
    input_tensor = tf.convert_ = [line.strip() for line in f.readlines()]
            class_names_loaded = []
            to_tensor(normalized_image_array); input_tensor = tf.expand_dims(input_tensor,for line in class_names_raw:
                parts = line.split(" ", 1)
                if axis=0)
    predictions_np = None
    try:
        predictions_output = model_layer len(parts) > 1 and parts[0].isdigit():
                    class_names_loaded.append(parts[1])
                else:
                    class_names_loaded.append(line)
        return model_arg(input_tensor)
        if isinstance(predictions_output, dict):
            if len(predictions_output) == 1: predictions_tensor = list(predictions_output.values())[0]
            elif 'outputs_layer, class_names_loaded
    except Exception as e:
        print(f"DEBUG: Erreur chargement modèle/labels: {e}")
        return model_layer, class_names_loaded

def' in predictions_output: predictions_tensor = predictions_output['outputs']
            elif 'output_0' predict_insect_saved_model(image_cv2, model_layer_arg, class_names_arg in predictions_output: predictions_tensor = predictions_output['output_0']
            else:
                key_found = None
                for key, value in predictions_output.items():
                    if isinstance(value, tf, input_size): # Renommer en predict_arthropod
    if model_layer_arg is None or class_names_arg is None:
        return "Erreur Modèle/Labels", 0.0.Tensor) and len(value.shape) == 2 and value.shape[0] == 1:
                        predictions_tensor = value; key_found = key; break
                if key_found is None:, []
    img_resized = cv2.resize(image_cv2, input_size, interpolation=cv2.INTER_AREA)
    image_array = np.asarray(img_resized, dtype=np return "Erreur Sortie Modèle Dict", 0.0, []
        else: predictions_tensor = predictions_output
        if hasattr(predictions_tensor, 'numpy'): predictions_np = predictions_tensor.numpy.float32)
    normalized_image_array = (image_array / 127.5()
        else: predictions_np = np.array(predictions_tensor)
    except Exception as e_) - 1.0
    input_tensor = tf.convert_to_tensor(normalized_image_array)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    predict: print(f"DEBUG: Erreur prédiction: {e_predict}"); return "Erreur Prédictionpredictions_np = None
    try:
        predictions_output = model_layer_arg(input_tensor", 0.0, []
    if predictions_np is None or predictions_np.size == 0)
        if isinstance(predictions_output, dict):
            if len(predictions_output) == 1: return "Erreur Prédiction Vide", 0.0, []
    predicted_class_index = np: predictions_tensor = list(predictions_output.values())[0]
            elif 'outputs' in predictions.argmax(predictions_np[0]); confidence_score = predictions_np[0][predicted_class_index_output: predictions_tensor = predictions_output['outputs']
            elif 'output_0' in predictions_]
    if predicted_class_index >= len(class_names_arg): return "Erreur Index Labeloutput: predictions_tensor = predictions_output['output_0']
            else:
                key_found =", confidence_score, predictions_np[0]
    label_name = class_names_arg[predicted_class None
                for key, value in predictions_output.items():
                    if isinstance(value, tf.Tensor_index]
    return label_name, confidence_score, predictions_np[0]

def create_) and len(value.shape) == 2 and value.shape[0] == 1:
                        label_display_image(label_image_data, filtered_props):
    if label_image_data.predictions_tensor = value; key_found = key; break
                if key_found is None: return "ndim == 3 and label_image_data.shape[2] == 1: label_image_dataErreur Sortie Modèle Dict", 0.0, []
        else: predictions_tensor = predictions_output
         = label_image_data.squeeze(axis=2)
    elif label_image_data.ndim !=if hasattr(predictions_tensor, 'numpy'): predictions_np = predictions_tensor.numpy()
        else: 2:
        h, w = (200,200) if not filtered_props or not predictions_np = np.array(predictions_tensor)
    except Exception as e_predict: print(f hasattr(filtered_props[0],'image') else filtered_props[0].image.shape[:2]
"DEBUG: Erreur prédiction: {e_predict}"); return "Erreur Prédiction", 0.0        return np.zeros((h, w, 3), dtype=np.uint8)
    label_, []
    if predictions_np is None or predictions_np.size == 0: return "Erreurdisplay = np.zeros((label_image_data.shape[0], label_image_data.shape[ Prédiction Vide", 0.0, []
    predicted_class_index = np.argmax(predictions_1], 3), dtype=np.uint8)
    for prop_item in filtered_props:
np[0])
    confidence_score = predictions_np[0][predicted_class_index]
            color = np.random.randint(50, 256, size=3)
        for coordif predicted_class_index >= len(class_names_arg): return "Erreur Index Label", confidence_score, in prop_item.coords:
            if 0 <= coord[0] < label_display.shape[ predictions_np[0]
    label_name = class_names_arg[predicted_class_index]0] and 0 <= coord[1] < label_display.shape[1]:
                label_display
    return label_name, confidence_score, predictions_np[0]


def create_label_display[coord[0], coord[1]] = color
    return label_display

def main():
    st_image(label_image_data, filtered_props):
    if label_image_data.ndim ==.set_page_config(layout="wide")
    st.title("Détection, isolation et identification d 3 and label_image_data.shape[2] == 1:
        label_image_dataʼArthropodes")

    if 'image_data_list' not in st.session_state: st = label_image_data.squeeze(axis=2)
    elif label_image_data.ndim !=.session_state.image_data_list = []
    if 'model_obj' not in st. 2:
        h, w = (200, 200) if not filtered_props or notsession_state: st.session_state.model_obj = None
    if 'class_names_list hasattr(filtered_props[0], 'image') else filtered_props[0].image.shape[:2]' not in st.session_state: st.session_state.class_names_list = None
    
        return np.zeros((h, w, 3), dtype=np.uint8)

    labelif 'active_image_id_for_params' not in st.session_state: st.session__display = np.zeros((label_image_data.shape[0], label_image_data.shapestate.active_image_id_for_params = None
    if 'first_model_load_message[1], 3), dtype=np.uint8)
    for prop_item in filtered_props:' not in st.session_state: st.session_state.first_model_load_message = False
        color = np.random.randint(50, 256, size=3)
        

    if st.session_state.model_obj is None:
        model_loaded, class_namesfor coord in prop_item.coords:
            if 0 <= coord[0] < label_display._loaded = load_saved_model_and_labels(SAVED_MODEL_DIR_PATH, LABELS_PATHshape[0] and 0 <= coord[1] < label_display.shape[1]:
                label)
        if model_loaded:
            st.session_state.model_obj = model_loaded
_display[coord[0], coord[1]] = color
    return label_display

def main():
            if class_names_loaded: st.session_state.class_names_list = class_names_loaded    st.set_page_config(layout="wide")
    st.title("Détection, isolation et
            if not st.session_state.first_model_load_message and model_loaded and class_names_ identification dʼArthropodes")

    if 'image_data_list' not in st.session_state: st.session_state.image_data_list = []
    if 'model_obj' not in st.sessionloaded:
                st.sidebar.success("Modèle et labels chargés !") # Message dans la sidebar
                st_state: st.session_state.model_obj = None
    if 'class_names_list'.session_state.first_model_load_message = True
        # else: st.sidebar.error(" not in st.session_state: st.session_state.class_names_list = None
    ifÉchec chargement modèle.") # Optionnel

    model_to_use = st.session_state.model 'active_image_id_for_params' not in st.session_state: st.session_state.active_image_id_for_params = None
    if 'first_model_load_message'_obj
    class_names_to_use = st.session_state.class_names_list

 not in st.session_state: st.session_state.first_model_load_message = False

    with st.sidebar:
        st.header("Paramètres de Segmentation")
        active_id = st.session_    if st.session_state.model_obj is None:
        model_loaded, class_names_state.active_image_id_for_params
        active_img_data_sidebar = None
        if active_loaded = load_saved_model_and_labels(SAVED_MODEL_DIR_PATH, LABELS_id:
            try:
                active_img_data_sidebar = next(item for item in st.PATH)
        if model_loaded:
            st.session_state.model_obj = model_loadedsession_state.image_data_list if item["id"] == active_id)
            except StopIteration
            if class_names_loaded:
                st.session_state.class_names_list = class: st.session_state.active_image_id_for_params = None; active_id = None

        _names_loaded
                if not st.session_state.first_model_load_message:
                    if active_img_data_sidebar:
            st.markdown(f"**Pour : {active_img_datast.success("Modèle d'identification et labels chargés avec succès !")
                    st.session_state.first_sidebar['filename']}**")
            params_sb = active_img_data_sidebar["params"]
            #_model_load_message = True
            else: st.warning("Modèle chargé, mais échec du chargement des params_sb["target_insect_count"] = st.number_input("Arthropodes attendus",0 labels.")
    
    model_to_use = st.session_state.model_obj
    class,100,params_sb["target_insect_count"],1,key=f"sb_target_names_to_use = st.session_state.class_names_list

    with st.sidebar_{active_id}")
            params_sb["blur_kernel"]=st.slider("Flou (0:
        st.header("Paramètres de Segmentation")
        active_id = st.session_state.=aucun)",0,21,params_sb["blur_kernel"],1,key=f"sb_active_image_id_for_params
        active_img_data_sidebar = None
        if active_id:
            try:
                active_img_data_sidebar = next(item for item in st.blur_{active_id}")
            params_sb["adapt_block_size"]=st.slider("Blocsession_state.image_data_list if item["id"] == active_id)
            except StopIteration Adapt.",3,51,params_sb["adapt_block_size"],2,key=f"sb_block_{:
                st.session_state.active_image_id_for_params = None; active_idactive_id}")
            params_sb["adapt_c"]=st.slider("Constante C",- = None

        if active_img_data_sidebar:
            st.markdown(f"**Pour : {active20,20,params_sb["adapt_c"],1,key=f"sb_c_{active_id}")
            params_sb["min_area"]=st.slider("Aire Min",10_img_data_sidebar['filename']}**")
            params_sidebar_ref = active_img_data,10000,params_sb["min_area"],10,key=f"sb__sidebar["params"]
            # params_sidebar_ref["target_insect_count"] = st.number_input("area_{active_id}")
            params_sb["morph_kernel"]=st.slider("NoyauArthropodes attendus (info)", 0,100, params_sidebar_ref["target_insect_count Morpho",1,15,params_sb["morph_kernel"],2,key=f"sb_"],1, key=f"sb_target_{active_id}")
            params_sidebar_ref["blurmorph_k_{active_id}")
            params_sb["morph_iterations"]=st.slider("It_kernel"] = st.slider("Flou (0=aucun)", 0, 21, params. Morpho",1,5,params_sb["morph_iterations"],1,key=f"sb__sidebar_ref["blur_kernel"], 1, key=f"sb_blur_{active_id}")morph_i_{active_id}")
            params_sb["margin"]=st.slider("Marge Ext
            params_sidebar_ref["adapt_block_size"] = st.slider("Bloc Adapt.", 3.",0,50,params_sb["margin"],key=f"sb_margin_{active_id}")
            , 51, params_sidebar_ref["adapt_block_size"], 2, key=f"params_sb["use_circularity"]=st.checkbox("Filtre Circ.",params_sb["usesb_block_{active_id}")
            params_sidebar_ref["adapt_c"] = st.slider_circularity"],key=f"sb_circ_c_{active_id}")
            if params_sb("Constante C", -20, 20, params_sidebar_ref["adapt_c"], ["use_circularity"]:
                params_sb["min_circularity"]=st.slider("Circ. Min Val",1, key=f"sb_c_{active_id}")
            params_sidebar_ref["min_0.0,1.0,params_sb["min_circularity"],0.05,key=area"] = st.slider("Aire Min", 10, 10000, params_sidebar_ref["min_area"], 10, key=f"sb_area_{active_id}")
f"sb_circ_v_{active_id}")
            params_sb["apply_relative_filter"]=st.checkbox("Filtre Relatif",params_sb["apply_relative_filter"],key=f            params_sidebar_ref["morph_kernel"] = st.slider("Noyau Morpho", 1, 15, params_sidebar_ref["morph_kernel"], 2, key=f"sb_"sb_rel_f_{active_id}")

            if st.button("Appliquer Paramètres & Traiter Imagemorph_k_{active_id}")
            params_sidebar_ref["morph_iterations"] = st.slider Active", key=f"sb_apply_btn_global_{active_id}"):
                with st.spinner("It. Morpho", 1, 5, params_sidebar_ref["morph_iterations"], 1(f"Traitement de {active_img_data_sidebar['filename']}..."):
                    active_img_, key=f"sb_morph_i_{active_id}")
            params_sidebar_ref["margindata_sidebar["processed_data"] = process_image(active_img_data_sidebar["cv_image"] = st.slider("Marge Ext.", 0, 50, params_sidebar_ref["margin"], params_sb)
                    active_img_data_sidebar["is_processed"] = True
                st.rerun"], key=f"sb_margin_{active_id}")
            params_sidebar_ref["use_circular()
        else:
            st.info("Sélectionnez une image (bouton '⚙️ Configurer') pourity"] = st.checkbox("Filtre Circ.", params_sidebar_ref["use_circularity"], key ajuster ses paramètres ici.")

    tab1, tab2, tab3 = st.tabs(["Segmentation par Image",=f"sb_circ_c_{active_id}")
            if params_sidebar_ref["use_ "Analyse Globale", "Guide"])

    with tab1:
        st.header("Configuration et Segmentationcircularity"]:
                params_sidebar_ref["min_circularity"] = st.slider("Circ. Min Val", 0.0, 1.0, params_sidebar_ref["min_circularity"],  Image par Image")
        uploaded_files_tab1 = st.file_uploader("1. Choisissez vos images", type0.05, key=f"sb_circ_v_{active_id}")
            params_sidebar=["jpg","jpeg","png"], accept_multiple_files=True, key="tab1_file_uploader_main_ref["apply_relative_filter"] = st.checkbox("Filtre Relatif", params_sidebar_ref["apply_relative_filter"], key=f"sb_rel_f_{active_id}")

            _v3")
        
        if uploaded_files_tab1:
            # ... (Logique de mise à jour de st.session_state.image_data_list comme avant)
            new_ids_tab1 = {if st.button("Appliquer et Traiter l'Image Active", key=f"sb_apply_btn_v2_{active_id}"):
                with st.spinner(f"Traitement de {active_img_data_f.file_id + "_" + f.name for f in uploaded_files_tab1}
            existingsidebar['filename']}..."):
                    active_img_data_sidebar["processed_data"] = process_image_map_tab1 = {img_d["id"]: img_d for img_d in st.session_(
                        active_img_data_sidebar["cv_image"], 
                        params_sidebar_ref state.image_data_list}
            updated_list_tab1 = []
            changed_files_tab1 =
                    )
                    active_img_data_sidebar["is_processed"] = True
                    st.rerun() 
        else:
            st.info("Sélectionnez une image (bouton '⚙️ Configurer') False
            for up_file in uploaded_files_tab1:
                img_id_up = up pour ajuster ses paramètres ici.")

    tab1, tab2, tab3 = st.tabs(["Segmentation par_file.file_id + "_" + up_file.name
                if img_id_up in existing_map_tab1: updated_list_tab1.append(existing_map_tab1[img_id_up Image", "Analyse Globale", "Guide"])

    with tab1:
        st.header("Configuration et Segmentation])
                else:
                    changed_files_tab1 = True
                    img_bytes_up = up Image par Image")
        uploaded_files_tab1 = st.file_uploader(
            "1. Choisissez vos_file.getvalue(); img_cv_up = cv2.imdecode(np.frombuffer(img_ images", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="bytes_up, np.uint8), cv2.IMREAD_COLOR)
                    updated_list_tabtab1_main_file_uploader_key_v3"
        )
        if uploaded_files_tab11.append({"id": img_id_up, "filename": up_file.name, "image_bytes": img:
            new_ids_tab1 = {f.file_id + "_" + f.name for f in uploaded__bytes_up, 
                                            "cv_image": img_cv_up, "params": DEFAULT_SEG_PARAMS.copy(), 
                                            "processed_data": None, "is_processed": False})
            iffiles_tab1}
            existing_map_tab1 = {item["id"]: item for item in st len(updated_list_tab1) != len(st.session_state.image_data_list):.session_state.image_data_list}
            updated_list_tab1 = []
            changed changed_files_tab1 = True
            st.session_state.image_data_list = updated_list_files_flag_tab1 = False
            for up_file_tab1 in uploaded_files_tab_tab1
            if changed_files_tab1:
                if st.session_state.image_1:
                file_id_tab1 = up_file_tab1.file_id + "_" + up_data_list: st.session_state.active_image_id_for_params = st.session_file_tab1.name
                if file_id_tab1 in existing_map_tab1:
state.image_data_list[0]["id"]
                else: st.session_state.active_                    updated_list_tab1.append(existing_map_tab1[file_id_tab1])image_id_for_params = None
                st.rerun()

        if not st.session_
                else:
                    changed_files_flag_tab1 = True
                    bytes_tab1 = upstate.image_data_list: st.info("Veuillez téléverser des images.")
        
        if st.session_state.image_data_list:
            st.markdown("---")
            _file_tab1.getvalue(); cv_img_tab1 = cv2.imdecode(np.frombuffer(bytes_tab1, np.uint8), cv2.IMREAD_COLOR)
                    updated_list_tabif st.button("Traiter TOUTES les images (avec leurs paramètres respectifs)", key="process_all_1.append({"id": file_id_tab1, "filename": up_file_tab1.name,tab1_v3"):
                # ... (Logique du bouton Traiter TOUTES comme avant)
                 "image_bytes": bytes_tab1, 
                                           "cv_image": cv_img_tab1,num_all = len(st.session_state.image_data_list); prog_all = st.progress "params": DEFAULT_SEG_PARAMS.copy(), 
                                           "processed_data": None, "is(0); stat_all = st.empty()
                for i_all_proc, img_d_all__processed": False})
            if len(updated_list_tab1) != len(st.session_proc in enumerate(st.session_state.image_data_list):
                    stat_all.text(f"Traitstate.image_data_list): changed_files_flag_tab1 = True
            st.session_state.image_data_list = updated_list_tab1
            if changed_files_flag_tabement {img_d_all_proc['filename']} ({i_all_proc+1}/{num_all})...")
                    img_d_all_proc["processed_data"] = process_image(img_d1:
                st.session_state.active_image_id_for_params = st.session__all_proc["cv_image"], img_d_all_proc["params"])
                    img_d_allstate.image_data_list[0]["id"] if st.session_state.image_data_list else None
                st.rerun()

        if not st.session_state.image_data_list_proc["is_processed"] = True
                    prog_all.progress((i_all_proc+1)/: st.info("Veuillez téléverser des images.")
        
        if st.session_state.imagenum_all)
                stat_all.success("Toutes images traitées.")

        for idx_main_data_list:
            st.markdown("---")
            if st.button("Traiter TOUTES, img_data_main in enumerate(st.session_state.image_data_list):
            st les images (avec leurs paramètres respectifs)", key="process_all_btn_tab1_v3"):
                num.markdown(f"--- \n ### Image {idx_main + 1}: {img_data_main['filename_all = len(st.session_state.image_data_list); prog_all = st.progress']}")
            select_btn_key_main = f"select_cfg_btn_main_{img_data_main(0); stat_all = st.empty()
                for i_all_p, img_d_p_['id']}"
            col_img_main, col_actions_main = st.columns([4,1])
all in enumerate(st.session_state.image_data_list):
                    stat_all.text(f"Traitement de {img_d_p_all['filename']} ({i_all_p+1}/{num            with col_img_main:
                 st.image(cv2.cvtColor(img_data_main["cv_all})...")
                    img_d_p_all["processed_data"] = process_image(_image"], cv2.COLOR_BGR2RGB), caption="Originale", use_column_width=True)
            img_d_p_all["cv_image"], img_d_p_all["params"])
                    with col_actions_main:
                # MODIFICATION : Bouton "Configurer" plus grand et clair
                if stimg_d_p_all["is_processed"] = True
                    prog_all.progress((i_all_p + 1) / num_all)
                stat_all.success("Toutes les images ont été trait.button(f"⚙️ Configurer cette Image", key=select_btn_key_main, help=f"Éditer les paramètres pour {img_data_main['filename']}", use_container_width=True):
ées.")

        for idx_main_tab1, img_data_main_tab1 in enumerate(st                    st.session_state.active_image_id_for_params = img_data_main["id.session_state.image_data_list):
            st.markdown(f"--- \n ### Image {idx_"]
                    st.info(f"'{img_data_main['filename']}' sélectionnée. Paramètres modmain_tab1 + 1}: {img_data_main_tab1['filename']}")
            select_btnifiables dans la barre latérale.")
                    # st.rerun() # Optionnel, dépend de la ré_key_main_tab1 = f"select_sidebar_btn_main_tab1_{img_dataactivité de la sidebar

            if img_data_main["is_processed"] and img_data_main["_main_tab1['id']}"
            col_img_main_tab1, col_action_mainprocessed_data"]:
                cols_res_main = st.columns(2)
                with cols_res__tab1 = st.columns([4,1])
            with col_img_main_tab1:
                 main[0]:
                    st.image(img_data_main["processed_data"]["opening"], channels="GRAY", captionst.image(cv2.cvtColor(img_data_main_tab1["cv_image"], cv2="Résultat Morphologique", use_column_width=True)
                with cols_res_main[1.COLOR_BGR2RGB), caption="Originale", use_column_width=True)
            with col_action_]:
                    label_disp_img_main = create_label_display_image(img_data_mainmain_tab1:
                if st.button(f"⚙️ Configurer", key=select_btn["processed_data"]["labels"], img_data_main["processed_data"]["filtered_props"])
                    num_key_main_tab1, help=f"Éditer params pour {img_data_main__detected_main = len(img_data_main['processed_data']['filtered_props'])
                    sttab1['filename']}"):
                    st.session_state.active_image_id_for_params = img.image(label_disp_img_main, caption=f"Arthropodes Détectés: {num_data_main_tab1["id"]
                    st.info(f"'{img_data_main_detected_main}", use_column_width=True)
                st.metric(label=f"Arthropodes Détectés", value=num_detected_main)
            else: st.caption("Résultats de segmentation appara_tab1['filename']}' sélectionnée. Modifiez ses paramètres dans la barre latérale.")
                    # st.rerun() # Pour forcer la mise à jour de la sidebar immédiatement

            if img_data_main_tab1îtront ici après traitement.")

    with tab2:
        st.header("Analyse Globale des Arthropodes Ident["is_processed"] and img_data_main_tab1["processed_data"]:
                cols_resifiés")
        if model_to_use is None or class_names_to_use is None:
            st.error("Modèle d'identification ou labels non disponibles.")
        elif not st.session_state.image_main_tab1 = st.columns(2)
                with cols_res_main_tab1[0]:
_data_list or not any(img_d["is_processed"] for img_d in st.session                    st.image(img_data_main_tab1["processed_data"]["opening"], channels="GRAY", caption="_state.image_data_list):
            st.info("Veuillez d'abord traiter des images (Résultat Morphologique", use_column_width=True)
                with cols_res_main_tab1[1]:
                    label_disp_main_tab1 = create_label_display_image(img_Onglet 'Segmentation par Image').")
        else:
            # ... (Logique de collecte et affichage dudata_main_tab1["processed_data"]["labels"], img_data_main_tab1["processed_ Pie Chart et Shannon)
            all_labels_pie_tab2 = []
            processed_imgs_tab2 = [data"]["filtered_props"])
                    num_det_main_tab1 = len(img_data_main_tab1['processed_data']['filtered_props'])
                    st.image(label_disp_main_tab1,img for img in st.session_state.image_data_list if img["is_processed"] and img["processed_data"]]
            if not processed_imgs_tab2: st.info("Aucune image traitée pour caption=f"Arthropodes Détectés: {num_det_main_tab1}", use_column identification.")
            else:
                st.write(f"Analyse basée sur {len(processed_imgs_width=True)
                st.metric(label=f"Arthropodes Détectés", value=num_det_main_tab1)
            else: st.caption("Résultats de segmentation apparaîtront ici après traitement.")

    _tab2)} image(s) traitée(s).")
                for img_d_item_tab2 in processedwith tab2:
        st.header("Analyse Globale des Arthropodes Identifiés")
        if_imgs_tab2:
                    if not (img_d_item_tab2.get("cv_image") is not None and 
                            img_d_item_tab2.get("processed_data", {} model_to_use is None or class_names_to_use is None:
            st.error(").get("filtered_props") is not None and
                            img_d_item_tab2.get("Modèle d'identification ou labels non disponibles.")
        elif not st.session_state.image_data_list or not any(img_d_tab2_an["is_processed"] for img_d_tabparams", {}).get("margin") is not None):
                        continue
                    extracted_arth_tab2 = extract2_an in st.session_state.image_data_list):
            st.info("Veu_insects(img_d_item_tab2["cv_image"], img_d_item_tab2["processed_data"]["filtered_props"], img_d_item_tab2["params"]["margin"])
illez d'abord téléverser et traiter des images.")
        else:
            all_labels_pie_tab2_                    for arth_item_tab2 in extracted_arth_tab2:
                        label_val_tab2, _,an = []
            imgs_processed_tab2_an = [img_tab2_an for img_ _ = predict_insect_saved_model(arth_item_tab2["image"], model_to_use, class_tab2_an in st.session_state.image_data_list if img_tab2_an["is_processed"] and img_tab2_an["processed_data"]]
            if not imgs_processed_tabnames_to_use, MODEL_INPUT_SIZE)
                        if "Erreur" not in label_val_2_an: st.info("Aucune image traitée pour l'identification.")
            else:
                st.writetab2: all_labels_pie_tab2.append(label_val_tab2)
                
(f"Analyse basée sur {len(imgs_processed_tab2_an)} image(s) trait                if not all_labels_pie_tab2: st.warning("Aucun arthropode identifié.")
                else:ée(s).")
                for img_item_tab2_an in imgs_processed_tab2_an:
                    eco_counts_tab2 = {}; non_mapped_tab2 = set()
                    for lbl_pie
                    if not (img_item_tab2_an.get("cv_image") is not None and 
_tab2 in all_labels_pie_tab2:
                        map_key = "Opiliones et Araneae" if lbl_pie_tab2 == "Arachnides" else lbl_pie_tab2                            img_item_tab2_an.get("processed_data", {}).get("filtered_props") is not None and
                            img_item_tab2_an.get("params", {}).get("margin")
                        if map_key not in ECOLOGICAL_FUNCTIONS_MAP: non_mapped_tab2.add( is not None):
                        st.warning(f"Données manquantes pour {img_item_tab2_an.lbl_pie_tab2)
                        eco_func = ECOLOGICAL_FUNCTIONS_MAP.get(map_get('filename', 'une image')}, ignorée."); continue
                    
                    extracted_arthropods_tabkey, DEFAULT_ECOLOGICAL_FUNCTION)
                        eco_counts_tab2[eco_func] =2_an = extract_insects(
                        img_item_tab2_an["cv_image"], img_ eco_counts_tab2.get(eco_func, 0) + 1
                    if non_mapped_tabitem_tab2_an["processed_data"]["filtered_props"], 
                        img_item_tab22: st.warning(f"Labels non mappés: {non_mapped_tab2}")

                    if eco__an["params"]["margin"]
                    )
                    for arthropod_tab2_an in extracted_arthropcounts_tab2:
                        # MODIFICATION: Mise en page avec tableau à gauche, pie chart à droite
ods_tab2_an:
                        label_val_tab2_an, _, _ = predict_insect_saved_                        col_table_stats, col_pie_chart = st.columns([3, 2]) # 3 partsmodel(
                            arthropod_tab2_an["image"], model_to_use, class_names pour table, 2 pour pie

                        with col_table_stats:
                            st.subheader("Récap_to_use, MODEL_INPUT_SIZE
                        )
                        if "Erreur" not in label_val_itulatif par Groupe et Fonction")
                            # Créer les données pour le DataFrame
                            table_data = []tab2_an: all_labels_pie_tab2_an.append(label_val_tab2_an
                            # D'abord, compter les labels bruts
                            raw_label_counts = {}
                            for l)
                
                if not all_labels_pie_tab2_an: st.warning("Aucun in all_labels_pie_tab2:
                                raw_label_counts[l] = raw_label_ arthropode n'a pu être identifié.")
                else:
                    # --- Création du tableau et du pie chart côte à côte ---
                    col_table_summary, col_pie_chart_display = st.counts.get(l, 0) + 1
                            
                            for label_raw, qty in sortedcolumns([2,1]) # 2 parts pour le tableau, 1 pour le pie chart
                    
                    with(raw_label_counts.items()):
                                display_label = "Opiliones et Araneae" col_table_summary:
                        st.subheader("Résumé des Identifications")
                        # Compter les labels if label_raw == "Arachnides" else label_raw
                                func = ECOLOGICAL_FUNCTIONS_MAP.get(display_label, DEFAULT_ECOLOGICAL_FUNCTION)
                                table_data.append({" bruts et leurs fonctions
                        raw_label_counts = {}
                        for lbl in all_labels_pie_tabGroupe Identifié": display_label, "Quantité": qty, "Fonction Écologique": func})
                            2_an:
                            raw_label_counts[lbl] = raw_label_counts.get(lbl, 0
                            if table_data:
                                df_recap = pd.DataFrame(table_data)
                               ) + 1
                        
                        summary_data = []
                        for label_name, count in sorted(raw_label st.dataframe(df_recap, use_container_width=True)
                            
                            shannon__counts.items(), key=lambda item: item[1], reverse=True):
                            # Gérer leval_tab2 = calculate_shannon_index(eco_counts_tab2)
                            st.subheader(" renommage pour l'affichage
                            display_label_name = "Opiliones et Araneae"Indice de Shannon Fonctionnel (H')")
                            st.metric(label="H' Global", value=f if label_name == "Arachnides" else label_name
                            eco_func = ECOLOGICAL"{shannon_val_tab2:.3f}")
                            if shannon_val_tab2 == _FUNCTIONS_MAP.get(display_label_name, ECOLOGICAL_FUNCTIONS_MAP.get(label_0 and sum(eco_counts_tab2.values()) > 0: st.caption("Une seule fonction écologique présente.")
                            elif shannon_val_tab2 > 0:
                                max_s = math.logname, DEFAULT_ECOLOGICAL_FUNCTION))
                            summary_data.append({
                                "Groupe Taxonomique(len(eco_counts_tab2)) if len(eco_counts_tab2) > 0 else": display_label_name,
                                "Quantité": count,
                                "Fonction Écologique": eco 0
                                st.caption(f"Max H' pour {len(eco_counts_tab2)} fonctions:_func
                            })
                        df_summary = pd.DataFrame(summary_data)
                        st.dataframe(df {max_s:.3f}.")

                        with col_pie_chart:
                            st.subheader("Fon_summary, use_container_width=True)

                        # Calcul de Shannon sur les fonctions écologiques
                        ecological_ctions Écologiques")
                            pie_keys = list(eco_counts_tab2.keys()); pie_sizes = list(eco_counts_tab2.values())
                            colors_map_pie_val = {"Décounts_for_shannon = {}
                        for data_row in summary_data: # Utiliser les données déjàcomposeurs et ingénieurs du sol": "#8B4513", "Pollinisateurs": "#FF préparées
                            func = data_row["Fonction Écologique"]
                            ecological_counts_for_shD700", "Prédateurs": "#DC143C", "Ravageur": "#FF8C0annon[func] = ecological_counts_for_shannon.get(func, 0) + data_row["Quant0", "Non défini": "#D3D3D3"}
                            pie_colors_val = [colors_ité"]
                        
                        if ecological_counts_for_shannon:
                            shannon_val_tab2_map_pie_val.get(k, "#CCCCCC") for k in pie_keys]
                            
                            #an = calculate_shannon_index(ecological_counts_for_shannon)
                            st.metric(label="Ind Taille du Pie Chart très réduite
                            fig_pie_final, ax_pie_final = plt.subplots(figsizeice de Shannon Fonctionnel Global (H')", value=f"{shannon_val_tab2_an:.3=(3, 2.1)) # Ajustez au besoin
                            ax_pie_final.pie(pief}")
                            # ... (captions Shannon)
                            if shannon_val_tab2_an == _sizes, labels=pie_keys, autopct='%1.1f%%', startangle=90, colors=0 and sum(ecological_counts_for_shannon.values()) > 0:
                                st.captionpie_colors_val, textprops={'fontsize': 5}) # Police très petite
                            ax_pie_final("H'=0: une seule fonction écologique présente.")
                            elif shannon_val_tab2_an > 0.axis('equal')
                            st.pyplot(fig_pie_final)
                    else: st.write("Auc:
                                max_s = math.log(len(ecological_counts_for_shannon)) if len(ecologicalune fonction écologique à afficher.")
            
            st.markdown("--- \n ### Identification Détaillée par Image")
            _counts_for_shannon) > 0 else 0
                                st.caption(f"Max H# ... (Affichage détaillé par image comme avant)
            for idx_det_tab2, img_d' pour {len(ecological_counts_for_shannon)} fonctions: {max_s:.3f}.")
                        _det_tab2 in enumerate(processed_imgs_tab2):
                st.markdown(f"#### {else:
                            st.caption("Aucune donnée pour l'indice de Shannon.")


                    with col_pieimg_d_det_tab2['filename']}")
                if not (img_d_det_tab2_chart_display:
                        ecological_counts_for_pie_chart = ecological_counts_for_shannon #.get("cv_image") is not None and 
                        img_d_det_tab2.get("processed_data", {}).get("filtered_props") is not None and
                        img_d_det_tab Réutiliser les comptes
                        if ecological_counts_for_pie_chart:
                            st.subheader("Fonctions Écologiques")
                            labels_pie_keys = list(ecological_counts_for_pie_chart2.get("params", {}).get("margin") is not None):
                    st.write("Données de segmentation incompl.keys())
                            sizes_pie_values = list(ecological_counts_for_pie_chart.valuesètes."); continue
                extracted_arth_det_tab2 = extract_insects(img_d_det_tab2["cv_image"], img_d_det_tab2["processed_data"]["filtered_props"], img_d_det())
                            # MODIFICATION: Couleurs pour le pie chart, s'assurer que la nouvelle clé est là
                            colors__tab2["params"]["margin"])
                if not extracted_arth_det_tab2: st.write("Aucunmap_pie = {"Décomposeurs et Ingénieurs du sol": "#8B4513", "Pollinisateurs": "#FFD700", 
                                              "Prédateurs": "#DC143C", arthropode extrait."); continue
                cols_det_tab2 = st.columns(3)
                for i "Ravageur": "#FF8C00", "Non défini": "#D3D3D3"}_arth_det, arth_det_item_tab2 in enumerate(extracted_arth_det_tab2):
                    
                            pie_colors_list = [colors_map_pie.get(lbl_p, "#CCCCCC") for lbl_lbl_det_tab2, conf_det_tab2, _ = predict_insect_saved_model(arth_p in labels_pie_keys]
                            
                            # MODIFICATION : Taille du Pie Chart très réduite
                            figdet_item_tab2["image"], model_to_use, class_names_to_use, MODEL_INPUT_pie, ax_pie = plt.subplots(figsize=(2.5, 1.75)) #_SIZE)
                    with cols_det_tab2[i_arth_det % 3]:
                         ex: 2.5 de large, 1.75 de haut
                            ax_pie.pie(sizesst.image(cv2.cvtColor(arth_det_item_tab2["image"], cv2.COLOR_BGR2RGB), caption=f"Arthropode #{arth_det_item_tab2['index']_pie_values, labels=None, autopct='%1.0f%%', startangle=90, 
                                       colors=pie_colors_list, pctdistance=0.8, textprops={'fontsize': 5 + 1}", width=150)
                        if "Erreur" in lbl_det_tab2: st.error(f"{lbl_det_tab2} ({conf_det_tab2*100:.2})
                            ax_pie.axis('equal')
                            # Légende séparée pour plus de clarté avecf}%)")
                        else:
                            lbl_disp_det = "Opiliones et Araneae" if un petit graphique
                            # plt.legend(labels_pie_keys, loc="center left", bbox_to_anchor lbl_det_tab2 == "Arachnides" else lbl_det_tab2
                            st.=(1, 0, 0.5, 1), fontsize='xx-small')
                            # plt.markdown(f"**Label:** {lbl_disp_det}")
                            st.markdown(f"**Fonction:** {ECOLOGICAL_FUNCTIONS_MAP.get(lbl_disp_det, ECOLOGICAL_tight_layout() # Pour que la légende ne sorte pas (peut ne pas suffire)
                            st.pyplot(figFUNCTIONS_MAP.get(lbl_det_tab2, DEFAULT_ECOLOGICAL_FUNCTION))}")
                            st.markdown(f"**Confiance:** {conf_det_tab2*100:.2_pie)
                            # Afficher la légende manuellement si besoin car elle peut être coupée sur un petit graphique
f}%")
                st.markdown("---")


    with tab3:
        st.header("Guide                            legend_html = "<div style='font-size: xx-small;'>"
                            for label_leg dʼutilisation")
        # ... (Guide mis à jour comme dans la version précédente)
        st.subheader("Segmentation, color_leg in zip(labels_pie_keys, pie_colors_list):
                                legend_html += f par Image (Onglet 1)")
        st.write("""
        1.  **Téléversez vos"<span style='color:{color_leg};'>■</span> {label_leg}<br>"
                            legend_html += " images.**
        2.  Pour chaque image, un bouton "⚙️ Configurer cette Image" apparaît. Cliquez dessus pour</div>"
                            st.markdown(legend_html, unsafe_allow_html=True)

                        else: st rendre cette image 'active'.
        3.  Les paramètres de l'image active s'affichent et peuvent être mod.write("Aucune fonction écologique à afficher.")
            
            st.markdown("--- \n ### Identification Détaillée parifiés dans la **barre latérale de gauche**.
        4.  Dans la sidebar, cliquez sur **" Image")
            # ... (Affichage détaillé par image comme avant)
            for idx_detail_tab2_Appliquer Paramètres & Traiter Image Active"** pour traiter l'image sélectionnée.
        5.  disp, img_data_item_detail_id_tab2_disp in enumerate(imgs_processed_tabUn bouton "Traiter TOUTES les images" est disponible pour lancer la segmentation sur l'ensemble du lot.
        """)
        st.subheader("Analyse Globale (Onglet 2)")
        st.write("""
        A2_an):
                st.markdown(f"#### {img_data_item_detail_id_tab2_disp['filename']}")
                if not (img_data_item_detail_id_tab2ffiche un tableau récapitulatif, un graphique des fonctions écologiques, l'Indice de Shannon, et l_disp.get("cv_image") is not None and 
                        img_data_item_detail_'identification détaillée.
        """)

    st.markdown("---")
    st.markdown("""
    **PSid_tab2_disp.get("processed_data", {}).get("filtered_props") is not None :** Quelques espèces de *Carabidae* (carabes) consomment des graines d'adventices, voire de and
                        img_data_item_detail_id_tab2_disp.get("params", {}).get("margin") is not None):
                    st.write("Données de segmentation incomplètes.")
                    continue semences agricoles (très marginalement). 
    Les isopodes peuvent aussi consommer de jeunes pousses (rare et faible impact).
    En cas de photos d'arthropodes en dehors des classes définies par le modèle
                extracted_arthropods_detail_id_tab2_disp = extract_insects(
                    img_data_item_detail_id_tab2_disp["cv_image"], 
                    img_data_item, l'outil renverra la classe qu'il considère comme étant la plus proche visuellement. 
    _detail_id_tab2_disp["processed_data"]["filtered_props"], 
                    img_dataAinsi, une photo d'*Andrenidae* pourrait être classée comme *Apidae*, bien que le modèle ait_item_detail_id_tab2_disp["params"]["margin"]
                )
                if not extracted été entraîné sur des photos d'*Apidae*.
    """)


if __name__ == "__main__":
    if '_arthropods_detail_id_tab2_disp:
                    st.write("Aucun arthropodeimage_data_list' not in st.session_state: st.session_state.image_data_ extrait pour identification sur cette image.")
                    continue
                num_cols_id_detail_disp_tab2_list = []
    if 'model_obj' not in st.session_state: st.session_state.model_obj = None
    if 'class_names_list' not in st.session_state:val = 3
                cols_id_detail_disp_tab2_val = st.columns(num_ st.session_state.class_names_list = None
    if 'active_image_id_forcols_id_detail_disp_tab2_val)
                col_idx_id_detail_disp_tab2_val = 0
                for arthropod_detail_item_id_tab2_disp_params' not in st.session_state: st.session_state.active_image_id_for_params = None
    if 'first_model_load_message' not in st.session_state: in extracted_arthropods_detail_id_tab2_disp:
                    label_detail_id_tab2_val, confidence_detail_id_tab2_val, _ = predict_insect_saved_ st.session_state.first_model_load_message = False
    main()

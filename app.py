import streamlit as st
from PIL import Image
from ultralytics import YOLO
import json
import os

# --------------------------------------
# Konfigurasi Halaman Streamlit
# --------------------------------------
st.set_page_config(page_title="FoodGenie 🍳", page_icon="🍲", layout="wide")
st.title("🍲 FoodGenie: Rekomendasi Resep dari Foto")

# --- Fungsi-fungsi Helper ---

@st.cache_resource
def load_model():
    """
    Memuat satu model AI utama.
    """
    # Ganti dengan path ke model terbaik Anda yang sudah dilatih
    model_path = "model_bahan_lengkap.pt" 

    if not os.path.exists(model_path):
        st.error(f"Model utama '{model_path}' tidak ditemukan. Menggunakan model default.")
        return YOLO("yolov8s.pt") # Fallback jika model utama tidak ada
        
    return YOLO(model_path)

@st.cache_data
def load_recipes():
    """Memuat data resep dari file JSON."""
    try:
        with open("recipes.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ File 'recipes.json' tidak ditemukan!")
        return []

def detect_ingredients(images, model):
    """
    Mendeteksi bahan dari BANYAK gambar menggunakan satu model.
    """
    all_detected_labels = set()

    for image in images:
        results = model(image)
        for r in results:
            for c in r.boxes.cls:
                label = model.names[int(c)]
                all_detected_labels.add(label.lower())
    
    return list(all_detected_labels)

def find_best_recipes(selected_ingredients, all_recipes):
    """Mencari dan mengurutkan resep berdasarkan bahan yang cocok."""
    if not selected_ingredients or not all_recipes: return []
    detected_set = set(selected_ingredients)
    matched_recipes = []
    for recipe in all_recipes:
        recipe_ingredients_set = set(bahan.lower() for bahan in recipe['bahan'])
        matching_count = len(detected_set.intersection(recipe_ingredients_set))
        if matching_count > 0:
            can_make = recipe_ingredients_set.issubset(detected_set)
            matched_recipes.append({
                "nama": recipe['nama'], "bahan": recipe['bahan'],
                "langkah": recipe['langkah'], "matching_count": matching_count,
                "can_make": can_make
            })
    return sorted(matched_recipes, key=lambda x: (x['can_make'], x['matching_count']), reverse=True)

# --- Antarmuka (UI) Streamlit ---

model = load_model()
all_recipes = load_recipes()

st.header("1. Unggah Foto Bahan")
uploaded_files = st.file_uploader(
    "Pilih satu atau beberapa gambar bahan makanan...", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    image_column, result_column = st.columns(2)
    with image_column:
        st.subheader("Gambar Terunggah")
        for uploaded_file in uploaded_files:
            st.image(Image.open(uploaded_file), caption=uploaded_file.name)

    with result_column:
        st.subheader("2. Hasil & Rekomendasi")
        if all_recipes:
            with st.spinner("🔍 Menganalisis gambar..."):
                images_to_process = [Image.open(file) for file in uploaded_files]
                # Memanggil fungsi deteksi yang telah disederhanakan
                detected_ingredients = detect_ingredients(images_to_process, model)
                
                if detected_ingredients:
                    st.info(f"**Bahan Terdeteksi:** {', '.join(d.capitalize() for d in detected_ingredients)}")
                else:
                    st.warning("**Tidak ada bahan yang dikenali.**")

                # Logika dropdown tidak berubah dan akan bekerja dengan benar
                recipe_ingredients = set()
                for recipe in all_recipes:
                    for ingredient in recipe['bahan']:
                        recipe_ingredients.add(ingredient.lower())

                all_possible_ingredients = sorted(list(set(detected_ingredients) | recipe_ingredients))
                
                selected_ingredients = st.multiselect(
                    "Koreksi bahan jika perlu (tambah/hapus):",
                    options=all_possible_ingredients,
                    default=detected_ingredients
                )
                
                if st.button("Cari Resep Sekarang!", type="primary", use_container_width=True) and selected_ingredients:
                    recommended_recipes = find_best_recipes(selected_ingredients, all_recipes)
                    st.subheader("📖 Rekomendasi Resep")
                    if not recommended_recipes:
                        st.warning("Tidak ada resep yang cocok dengan bahan yang Anda pilih.")
                    else:
                        for recipe in recommended_recipes:
                            if recipe['can_make']: status = "✅ Bisa Langsung Dibuat"
                            else: status = f"⚠️ Butuh Bahan Lain ({recipe['matching_count']}/{len(recipe['bahan'])} Cocok)"
                            with st.expander(f"{recipe['nama']} - {status}"):
                                st.markdown("**Bahan-bahan:**")
                                bahan_dimiliki = set(selected_ingredients)
                                for bahan in recipe['bahan']:
                                    if bahan.lower() in bahan_dimiliki: st.markdown(f"- {bahan.capitalize()} (✔️ Ada)")
                                    else: st.markdown(f"- <span style='color: red;'>{bahan.capitalize()} (❌ Tidak Ada)</span>", unsafe_allow_html=True)
                                st.markdown("\n**Langkah-langkah:**")
                                for i, step in enumerate(recipe['langkah'], 1): st.write(f"{i}. {step}")

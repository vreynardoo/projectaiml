import streamlit as st
from PIL import Image
from ultralytics import YOLO
import json
import tempfile
from collections import defaultdict

# --------------------------------------
# Load model YOLOv5 (Pindahkan ke atas!)
# --------------------------------------
# model = YOLO("yolov5s.pt")  # Menggunakan model pre-trained dari Ultralytics

# @st.cache_resource akan menyimpan model di cache, sehingga tidak perlu di-load ulang setiap kali ada interaksi.
@st.cache_resource
def load_model():
    """Memuat model YOLOv5 dari file."""
    model = YOLO("yolov5s.pt")  # Menggunakan model pre-trained dari Ultralytics
    return model

# @st.cache_data akan menyimpan data resep di cache.
@st.cache_data

# -------------------------
# Load resep dari file JSON
# -------------------------
def load_recipes():
    try:
        with open("data/recipes.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ File recipes.json tidak ditemukan di folder 'data/'!")
        return []
    except json.JSONDecodeError:
        st.error("❌ Format file 'data/recipes.json' salah! Pastikan formatnya adalah JSON yang valid.")
        return []

# -------------------------------
# Deteksi bahan makanan dari foto
# -------------------------------
def detect_ingredients(image, model):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        results = model(tmp.name)  # ← sekarang model sudah didefinisikan di atas

        detected_labels=set()
        for r in results:
            for c in r.boxes.cls:
                label = model.names[int(c)]
                detected_labels.add(label.lower())

    return list(detected_labels)


def find_best_recipes(detected_ingredients, all_recipes):
    # """
    # Mencari dan mengurutkan resep berdasarkan jumlah bahan yang cocok.
    # Resep dengan kecocokan bahan terbanyak akan muncul di paling atas.
    # """
    if not detect_ingredients or not all_recipes:
        return []
    
    detected_set=set(detected_ingredients)
    matched_recipes=[]

    for recipe in all_recipes:
        recipe_ingredients_set=set(recipe['bahan'])
        matching_count=len(detected_set.intersection(recipe_ingredients_set)) # Hitung berapa banyak bahan yang terdeteksi ada di dalam resep

        if matching_count > 0:
            can_make=recipe_ingredients_set.issubset(detected_set)
            matched_recipes.append({
                "nama": recipe['nama'],
                "bahan": recipe['bahan'],
                "langkah": recipe['langkah'],
                "matching_count": matching_count,
                "can_make": can_make #tandai apakah resep bisa langsung dibuat
            })

    # urutkan resep:
    # 1. Resep yang bisa dibuat (can_make = True) diutamakan.
    # 2. Urutkan berdasarkan jumlah bahan yang cocok (descending).

    return sorted(matched_recipes, key=lambda x:(c['can_make'], x['matching_count']), reverse=True) 


    

# --------------------
# Streamlit Web App UI
# --------------------
st.set_page_config(page_title="FoodGenie 🍳", page_icon="🍲", layout="wide")
st.title("🍲 FoodGenie: Rekomendasi Resep dari Foto Bahan Makanan")

model=load_model()
all_recipes=load_recipes()

col1, col2=st.column(2)

with col1:
    st.header("1. Unggah Foto Bahan")
    uploaded_file=st.file_uploader("Pilih gambar bahan makanan", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image=Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

with col2:
    st.header("2. Hasil & Rekomendasi")
    if uploaded_file and all_recipes:
        with st.spinner("🔍 Menganalisis gambar dan mencari resep..."):
            detected_ingredients=detect_ingredients(image, model) #deteksi bahan dari gambar
            st.info(f"**Bahan Terdeteksi:** {', '.join(detected_ingredients) if detected_ingredients else 'Tidak ada bahan yang dikenali.'}")

            selected_ingredients=st.multiselect(
                "Koreksi bahan jika perlu (tambah/hapus):",
                options=sorted(list(set(detected_ingredients + [bahan for resep in all_recipes for bahan in resep['bahan']]))),
                defaul=detected_ingredients
            )

            if st.button("Cari Resep", use_container_width=True) and selected_ingredients:
                recommended_recipes=find_best_recipes(selected_ingredients, all_recipes)

                st.subheader()




# uploaded_file = st.file_uploader("📸 Upload foto bahan makanan kamu", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     # Tampilkan gambar
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Gambar bahan makanan yang diunggah", use_column_width=True)

#     with st.spinner("🔍 Mendeteksi bahan makanan..."):
#         # Deteksi bahan
#         ingredients = detect_ingredients(image)
#         st.success(f"✅ Bahan terdeteksi: {', '.join(ingredients) if ingredients else 'Tidak ada bahan terdeteksi.'}")

#         # Load resep
#         all_recipes = load_recipes()

#         if ingredients:
#             st.header("📖 Rekomendasi Resep:")
#             shown = set()
#             for ingredient in ingredients:
#                 if ingredient in all_recipes:
#                     for resep in all_recipes[ingredient]:
#                         if resep["nama"] not in shown:
#                             shown.add(resep["nama"])
#                             st.subheader(f"🍽️ {resep['nama']}")
#                             st.markdown("**Bahan:**")
#                             st.write(", ".join(resep["bahan"]))
#                             st.markdown("**Langkah-langkah:**")
#                             for i, langkah in enumerate(resep["langkah"], 1):
#                                 st.write(f"{i}. {langkah}")
#                 else:
#                     st.info(f"ℹ️ Tidak ada resep ditemukan untuk bahan: {ingredient}")
#         else:
#             st.warning("⚠️ Tidak ada bahan yang dikenali dari gambar.")
# else:
#     st.info("📤 Silakan upload foto bahan makanan untuk mulai.")

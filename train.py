from ultralytics import YOLO
import os

def main():
    """
    Fungsi utama untuk melatih model YOLOv8 pada dataset BARU.
    """
    # Memulai dari model pre-trained 'yolov8s.pt' untuk hasil yang lebih baik dan cepat.
    model = YOLO('yolov8s.pt')

    print("Memulai pelatihan model AI dengan dataset 'AI Recipes Recommendation'...")
    
    try:
        # PERUBAHAN KUNCI: Path data sekarang menunjuk ke data.yaml dari dataset baru Anda.
        # Pastikan nama folder 'AI Recipes Recommendation' sudah benar.
        results = model.train(
            data='AI Recipes Recommendation/data.yaml', # <-- Ganti dengan nama folder dataset baru Anda
            epochs=25, 
            imgsz=640,
            # PERUBAHAN KUNCI: Memberi nama baru agar tidak menimpa hasil training sebelumnya.
            name='new_recipe_model_run' 
        )
        print("\nPelatihan berhasil diselesaikan!")
        print("Model terbaik Anda disimpan di folder: runs/detect/new_recipe_model_run/weights/best.pt")
    except Exception as e:
        print(f"\nTerjadi error selama pelatihan: {e}")
        print("Pastikan path ke file 'data.yaml' sudah benar dan sesuai dengan nama folder dataset Anda.")


if __name__ == '__main__':
    main()
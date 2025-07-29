import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model

# Modeli yükle
try:
    model = load_model("egzama_model.h5")
    print("Model başarıyla yüklendi!")
    print(f"Model katman sayısı: {len(model.layers)}")
    
    # Model yapısını göster
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name} - {type(layer).__name__}")
        
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    exit()

# Görselin yolu
img_path = "C:/Users/Gizem/Desktop/python_proje/veriler/Normal/normal100.png"
# Dosya kontrolü
if not os.path.exists(img_path):
    print(f"Görsel dosyası bulunamadı: {img_path}")
    exit()

# Görseli hazırla
try:
    img = cv2.imread(img_path)
    if img is None:
        print("Görsel okunamadı!")
        exit()
        
    img_original = img.copy()
    img = cv2.resize(img, (128, 128))
    img_array = img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    print(f"Görsel boyutu: {img_array.shape}")
    
except Exception as e:
    print(f"Görsel işleme hatası: {e}")
    exit()

# Convolutional katmanları bul
conv_layers = []
for i, layer in enumerate(model.layers):
    if 'conv' in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
        conv_layers.append((i, layer.name))
        
print(f"Bulunan conv katmanları: {conv_layers}")

if not conv_layers:
    print("Hiç convolutional katman bulunamadı!")
    exit()

# Son conv katmanını kullan (veya istediğiniz katmanı seçin)
last_conv_layer_idx = conv_layers[-1][0]
print(f"Kullanılacak katman: {last_conv_layer_idx} - {conv_layers[-1][1]}")

try:
    # Model'i build et (Sequential model için gerekli)
    if not model.built:
        model.build(input_shape=(None, 128, 128, 3))
    
    # İlk önce bir tahmin yap ki model build olsun
    dummy_prediction = model(img_array)
    print(f"Model build edildi. Dummy prediction shape: {dummy_prediction.shape}")
    
    # Sequential model için farklı yaklaşım
    # Direkt olarak model.layers kullanarak input tanımla
    last_conv_layer = model.layers[last_conv_layer_idx]
    
    # Yeni input tanımla
    inputs = Input(shape=(128, 128, 3))
    
    # Model katmanlarını manuel olarak bağla
    x = inputs
    for i, layer in enumerate(model.layers):
        x = layer(x)
        if i == last_conv_layer_idx:
            conv_outputs = x  # Bu katmanın çıktısını kaydet
    
    final_outputs = x
    
    # Grad-CAM modeli oluştur
    grad_model = Model(inputs=inputs, outputs=[conv_outputs, final_outputs])
    
    print("✓ Grad-CAM modeli başarıyla oluşturuldu!")
    
    # Gradient hesapla
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        # Sınıf sayısına göre ayarla
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Çok sınıflı model
            predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]
        else:
            # Binary model
            loss = predictions[:, 0]
    
    # Gradientleri hesapla
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        print("Gradient hesaplanamadı! Model trainable olmalı.")
        exit()
    
    # Grad-CAM hesapla
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    # Heatmap oluştur
    heatmap = tf.zeros(conv_outputs.shape[:2])
    for i in range(pooled_grads.shape[0]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]
    
    # Normalize et
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    print(f"Heatmap boyutu: {heatmap.shape}")
    print(f"Heatmap min/max: {heatmap.min()}/{heatmap.max()}")
    
    # Heatmap'i orijinal görsel boyutuna çevir
    heatmap_resized = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Colormap uygula
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Görselleri birleştir
    superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap_color, 0.4, 0)
    
    # Sonucu kaydet ve göster
    output_path = "gradcam_sonuc.jpg"
    cv2.imwrite(output_path, superimposed_img)
    print(f"Sonuç kaydedildi: {output_path}")
    
    # Matplotlib ile göster
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Orijinal görsel
    axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Orijinal Görsel")
    axes[0].axis("off")
    
    # Heatmap
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    # Birleştirilmiş sonuç
    axes[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Grad-CAM Sonucu")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Tahmin sonucunu da göster
    prediction = model.predict(img_array)[0]
    print(f"Model tahmini: {prediction}")
    
    # Başarı kontrolü
    print("\n=== GRAD-CAM BAŞARI KONTROLÜ ===")
    print(f"✓ Heatmap oluşturuldu: {heatmap.shape}")
    print(f"✓ Heatmap değer aralığı: {heatmap.min():.3f} - {heatmap.max():.3f}")
    print(f"✓ Sıfır olmayan piksel sayısı: {np.count_nonzero(heatmap)}")
    print(f"✓ Toplam piksel sayısı: {heatmap.size}")
    print(f"✓ Aktivasyon oranı: %{(np.count_nonzero(heatmap)/heatmap.size)*100:.1f}")
    
    if np.count_nonzero(heatmap) > heatmap.size * 0.01:  # En az %1 aktivasyon
        print("🎉 GRAD-CAM BAŞARILI! Anlamlı heatmap oluşturuldu.")
    else:
        print("⚠️  Heatmap çok zayıf - model belki de bu görsel için belirsiz.")
    
    if heatmap.max() > 0.1:
        print("✓ Güçlü aktivasyon bölgeleri tespit edildi.")
    else:
        print("⚠️  Zayıf aktivasyon - farklı bir katman deneyin.")
        
except Exception as e:
    print(f"Grad-CAM hesaplama hatası: {e}")
    import traceback
    traceback.print_exc()
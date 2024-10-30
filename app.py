import os
from flask import Flask, request, render_template, send_from_directory
import numpy as np
from keras.preprocessing import image
from keras.models import load_model  # Buraya ekleyin

# Flask uygulamasını başlat
app = Flask(__name__)

# Proje kök dizinini al
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Sınıf etiketleri (modeldeki sınıflarınıza göre düzenleyin)
classes = ['freshapples', 'freshbanana', 'freshcucumber', 'freshokra', 'freshoranges', 'freshpatato', 'freshtamto', 
            'rottenapples', 'rottenbanana', 'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpatato', 
            'rottentamto',]

# Ana sayfa (index) rotası
@app.route("/")
def index():
    return render_template("index.html")

# Yükleme işlemi için POST rotası
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')

    # 'images' dizini yoksa oluşturulur
    if not os.path.isdir(target):
        os.mkdir(target)

    # Yüklenen dosya işlenir
    for upload in request.files.getlist("file"):
        filename = upload.filename
        destination = os.path.join(target, filename)
        upload.save(destination)

        # Modeli yükle
        model = load_model('fruit_vegetable_model.keras')

        # Resmi işle
        test_image = image.load_img(destination, target_size=(256, 256))  # Modelinize uygun boyutu ayarlayın
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)  # Model için boyutu genişlet

        # Tahmin yap
        result = model.predict(test_image)
        predicted_class = np.argmax(result[0])  # En yüksek olasılığa sahip sınıfı seç
        prediction = classes[predicted_class]

    # Tahmin sonucunu ve resmi döndür
    return render_template("template.html", image_name=filename, text=prediction)

# Resmi görüntülemek için kullanılan rota
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

# Uygulamayı başlat
if __name__ == "__main__":
    app.run(debug=True)

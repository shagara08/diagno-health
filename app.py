# coding=utf-8
import os
import numpy as np

# Keras
from keras.models import load_model
from keras.utils import img_to_array
from keras.utils import load_img

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Mendefinisikan App Flask
app = Flask(__name__)

MODEL_PATH = 'models/SBPFixed_Model.h5'

# Memuat model yang sudah disimpan
model = load_model(MODEL_PATH)

# Membuat Fungsi Prediksi penyakit kulit


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = """
        <h2>Ini Adalah Cacar Air</h2>
        <h4>Cara Mengatasi Cacar Air:</h5>
        <ul>
            <li>1). Hindari makanan asin atau pedas</li>
            <li>2). Jangan menggaruk luka dan menjaga kuku tetap bersih</li>
            <li>3). Untuk menghindari gatal bisa menjadi parah, dengan memakai salep.</li>
            <li>4). Minum banyak cairan untuk mencegah dehidrasi,yang dapat menjadi komplikasi cacar air.</li>
            <li>5). Konsumsi obat penghilang rasa sakit untuk membantu mengurangi demam tinggi dan rasa sakit ketika seseorang menderita cacar air.</li>
        </ul>
        """
    elif preds == 1:
        preds = """
        <h2>Ini Adalah Herpes</h2>
        <h4>Cara Mengatasi Herpes:</h5>
        <ul>
            <li>1). Mengatur pola makan yang baik untuk mencegah penurunan daya tahan tubuh</li>
            <li>2). Kompres menggunakan air hangat atau dingin pada bagian yang sering muncul herpes untuk meredakan rasa sakit</li>
            <li>3). Mengonsumsi suplemen seperti yogurt, vitamin B dan zinc dengan takaran 30 mg per hari untuk mengatasi penyebaran virus </li>
            <li>4). Aplikasikan tumbukan halus bawang putih dan minyak zaitun pada bagian tubuh yang terdampak virus herpes tiga kali sehari </li>
            <li>5). Oleskan Cuka Apel ke bagian tubuh yang terdampak virus. Cuka apel memiliki komponen anti inflamasi yang bisa membuat luka cepat kering.</li>

        </ul>
        """
    elif preds == 2:
        preds = """
        <h2>ini Adalah Impetigo</h2>
        <h4>Cara Mengatasi Impetigo:</h5>
        <ul>
            <li>1). Gunakan salep atau krim antibiotik</li>
            <li>2). Merendam luka dengan menggunakan air hangat</li>
            <li>3). Meminum obat seperti clindamycin atau obat antibiotik golongan sefalosporin</li>
        </ul>
        """
    elif preds == 3:
        preds = """
        <h2>Ini Adalah Kurap</h2>
        <h4>Cara Mengatasi Kurap:</h5>
        <ul>
            <li>1). Keringkan area tubuh secara menyeluruh setelah mandi</li>
            <li>2). Gunakan pakaian longgar di daerah yang terkena kurap</li>
            <li>3). Cuci sprei dan pakaian setiap hari untuk membantu membunuh jamur-jamur</li>
            <li>4). Obati semua area yang terinfeksi dengan produk yang mengandung clotrimazole, miconazole, terbinafine, atau bahan terkait lainnya</li>
        </ul>
        """
    elif preds == 4:
        preds = """
        <h2>Ini Adalah Kutil</h2>
        <h4>Cara Mengatasi Kutil:</h5>
        <ul>
            <li>1). Perawatan laser</li>
            <li>2). Operasi pembedahan</li>
            <li>3). Perawatan dengan nitrogen cair/cryotherapy</li>
        </ul>
        """
    elif preds == 5:
        preds = """
        <h2>Ini Adalah Melanoma</h2>
        <h4>Cara Mengatasi Melanoma:</h5>
        <ul>
            <li>1). Kemoterapi</li>
            <li>2). Terapi radiasi</li>
            <li>3). Operasi atau pembedahan jadi pengobatan</li>
        </ul>
        """
    elif preds == 6:
        preds = """
        <h2>Ini Adalah Psoriasis</h2>
        <h4>Cara Mengatasi Psoriasis:</h5>
        <ul>
            <li>1). Membatasi waktu mandi</li>
            <li>2). Menggunakan bahan alami</li>
            <li>3). Menjalani pola makan sehat</li>
            <li>4). Mengelola stres dengan baik</li>
            <li>5). Mengoleskan pelembap pada kulit</li>
            <li>Mengenal dan menjauhi faktor pemicu gejala psoriasis</li>
        </ul>
        """
    elif preds == 7:
        preds = """
        <h2>Ini Adalah Vitiligo</h2>
        <h4>Cara Mengatasi Vitiligo</h5>
        <ul>
            <li>1). Operasi Cangkok kulit.</li>
            <li>2). Transplantasi suspensi seluler.</li>
            <li>3). Obat yang mengontrol peradangan.</li>
            <li>4). Pengobatan yang mempengaruhi sistem kekebalan.</li>
            <li>5). Terapi cahaya seperti Fototerapi dengan ultraviolet B pita sempit (UVB)</li>
        </ul>
        """

    return preds


@app.route('/', methods=['GET'])
def index():
    # Halaman Utama
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Mendapatkan file dari permintaan post
        f = request.files['file']

        # menyimpan file yang di upload ke folder images
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'images', secure_filename(f.filename))
        f.save(file_path)

        # Membuat prediksi
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)

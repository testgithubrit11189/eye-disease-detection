import os
from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
import cv2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dic = {
    0: 'cataract',
    1: 'diabetic_retinopathy',
    2: 'disc_edema',
    3: 'glaucoma',
    4: 'normal',
    5: 'retinal_detachment'
}

img_size_x = 128
img_size_y = 128

# ✅ LOAD TRAINED MODELS
cnn_model = load_model('model.h5')
vit_model = load_model('model.h5')


# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("home.html")


# =========================
# PREDICTION FUNCTION
# =========================
def predict_all(img_path):

    img = cv2.imread(img_path)

    # ---------- CNN ----------
    img_cnn = cv2.resize(img, (128, 128))
    img_cnn = img_cnn / 255.0
    img_cnn = np.reshape(img_cnn, (1, 128, 128, 3))

    cnn_pred = cnn_model.predict(img_cnn)
    cnn_conf = float(np.max(cnn_pred))
    cnn_class = dic[np.argmax(cnn_pred)]

    # ---------- MobileNet ----------
    img_vit = cv2.resize(img, (224, 224))
    img_vit = preprocess_input(img_vit)
    img_vit = np.reshape(img_vit, (1, 224, 224, 3))

    vit_pred = vit_model.predict(img_vit)
    vit_conf = float(np.max(vit_pred))
    vit_class = dic[np.argmax(vit_pred)]

    # ---------- 🔥 SMART FUSION ----------
    cnn_w = cnn_conf / (cnn_conf + vit_conf)
    vit_w = vit_conf / (cnn_conf + vit_conf)

    final_pred = (cnn_w * cnn_pred) + (vit_w * vit_pred)

    final_conf = float(np.max(final_pred)) * 100
    final_class = dic[np.argmax(final_pred)]

    return (
        cnn_class,round(cnn_conf*100,2),
        vit_class,round(vit_conf*100,2),
        final_class,round(final_conf,2)
    )
    


# =========================
# UPLOAD ROUTE
# =========================
@app.route("/upload", methods=["GET", "POST"])
def upload():

    cnn_p = vit_p = final_p = None
    cnn_conf = vit_conf = final_conf = None
    img_path = None

    if request.method == "POST" and 'photo' in request.files:

        file = request.files['photo']

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img_path = file_path

            cnn_p, cnn_conf, vit_p, vit_conf, final_p, final_conf = predict_all(img_path)

    return render_template(
        'upload.html',
        src=img_path,
        cnn_p=cnn_p,
        cnn_conf=cnn_conf,
        vit_p=vit_p,
        vit_conf=vit_conf,
        final_p=final_p,
        final_conf=final_conf
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)
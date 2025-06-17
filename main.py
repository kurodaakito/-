# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
import shutil
from base64 import b64encode
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# ==============
# FastAPI設定
# ==============
app = FastAPI()
index_file = Path("index.html")
upload_dir = Path("uploaded_images")
upload_dir.mkdir(exist_ok=True)  # ディレクトリがなければ作成

# ==============
# クラス定義(日本語)
# ==============
CLASSES_JP = [
    "一年生作物",     # AnnualCrop
    "森林",          # Forest
    "草地",          # HerbaceousVegetation
    "高速道路",      # Highway
    "工業地域",      # Industrial
    "牧草地",        # Pasture
    "多年生作物",    # PermanentCrop
    "住宅地",        # Residential
    "河川",          # River
    "湖沼・海"       # SeaLake
]

# ==============
# ViTモデルの構築
# ==============
model = torchvision.models.vit_b_16(weights=None)  # 空で初期化

# 最終層を出力 10 次元にあわせる
num_ftrs = model.heads[0].in_features
model.heads[0] = nn.Linear(num_ftrs, 10)

# 学習済みモデルの読み込み
model_path = Path("model.pth")  # ファインチューニング後に保存した重み
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()  # 推論モード

# 画像の前処理 (224x224, Normalize)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def get_index():
    """
    ルートパス (http://127.0.0.1:8000/) にアクセスしたときにindex.htmlを返す
    """
    return FileResponse(index_file)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    画像ファイルを受け取りViTモデルで推論し、
    画像と共にクラス+確率をHTMLで表示する
    """
    # ==============
    # 1) 画像保存
    # ==============
    image_path = upload_dir / file.filename
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # ==============
    # 2) 前処理
    # ==============
    img = Image.open(image_path).convert("RGB")  # 画像をRGBに変換
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)に次元調整
    
    # ==============
    # 3) 推論
    # ==============
    with torch.no_grad():
        outputs = model(img_tensor)             # shape=(1,10)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probabilities).item()
        pred_class = CLASSES_JP[pred_idx]
        pred_prob = probabilities[pred_idx].item()
    
    # ==============
    # 4) HTML表示
    # ==============
    # 画像をBase64文字列へ
    encoded = b64encode(image_path.read_bytes()).decode("utf-8")
    # index.htmlを読み込み
    original_html = index_file.read_text(encoding="utf-8")
    # {pred_class} や確率を表示
    # 例： <p>推定クラス: 森林 ( 確率 0.85 )</p>
    new_html = original_html.replace(
        '<p>No image uploaded yet.</p>',
        f"""
        <img src="data:image/jpeg;base64,{encoded}" style="max-width:500px;">
        <p>推定クラス: <strong>{pred_class}</strong> (確率: {pred_prob:.2f})</p>
        """
    )
    return HTMLResponse(content=new_html, status_code=200)
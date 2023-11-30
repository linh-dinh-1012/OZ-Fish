from fastapi import FastAPI, UploadFile
from ultralytics import YOLO

from PIL import Image
import json
from io import BytesIO


app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return [{"xmin":100,"ymin":100,"xmax":200,"ymax":200,"confidence":0.76,"class":12,"name":"poisson-clown"},
 {"xmin":25,"ymin":25,"xmax":50,"ymax":50,"confidence":0.76,"class":13,"name":"nemo"}]

@app.post("/upload/")
def get_bbox(file: UploadFile):
    model= YOLO("./best.pt")

    img= file.file.read()
    image =Image.open(BytesIO(img))
    results = model(image)
    res=results[0]
    boxes=[]
    for i in res.boxes:
        coor=i.xyxy[0]
        boxes.append({"xmin":coor[0].item()
                    ,"ymin":coor[1].item()
                    ,"xmax":coor[2].item()
                    ,"ymax":coor[-1].item()})

    # json_results = results_to_json(results,model)
    return boxes

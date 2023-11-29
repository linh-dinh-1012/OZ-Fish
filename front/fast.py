from fastapi import FastAPI

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {
        "box1":
        {"xmin":100,"ymin":100,"xmax":200,"ymax":200,"confidence":0.76,"class":12,"name":"poisson-clown"},
        "box2":
            {"xmin":25,"ymin":25,"xmax":50,"ymax":50,"confidence":0.76,"class":13,"name":"nemo"}
        }

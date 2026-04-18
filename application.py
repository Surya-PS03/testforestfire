from fastapi import FastAPI,Request,Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List
import pickle


### Application ###
application = FastAPI()

### Load Model ###
ridge = pickle.load(open(r"models/ridgeModel.pkl","rb"))
scaler = pickle.load(open(r"models/StandardScaler.pkl","rb"))

### templates ###
templates = Jinja2Templates(directory="templates")


### get home api ###
@application.get("/",response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse('index.html',
                                      {'request':request})

@application.post('/predict_fwi')
async def predict_fwi(request: Request,
    Temperature: float = Form(...),
    RH: float = Form(...),
    Ws: float = Form(...),
    Rain: float = Form(...),
    FFMC: float = Form(...),
    DMC: float = Form(...),
    DC: float = Form(...),
    BUI: float = Form(...),
    ISI: float = Form(...),
    Region: float = Form(...),
    Classes: float = Form(...)):

    features = [[Temperature,RH, Ws, Rain, FFMC, DMC,DC, ISI,BUI, Region, Classes]]
    scaled_features = scaler.transform(features)
    prediction = ridge.predict(scaled_features)[0]

    return templates.TemplateResponse("result.html", {"request":request,"prediction":prediction})


@application.get("/result", response_class=HTMLResponse)
async def result_page(request: Request):
    return templates.TemplateResponse("result.html", {"request": request, "prediction": "No prediction yet"})
from flask import Flask, request, jsonify  
import pandas as pd                        
import joblib                              
from typing import List, Dict, Any         

# ---- Konfiguration ----
model_filname = "student_model.joblib"        # Filnamn 

# ---- Ladda modellen en gång vid app-start ----
_loaded = joblib.load(model_filname)          # Ladda objektet från joblib-filen

# laddade är en bundle eller en pipeline
if isinstance(_loaded, dict) and "model" in _loaded:
    MODE = "bundle"                        
    BUNDLE = _loaded                       
    PIPE = None                            
else:
    MODE = "pipeline"                      
    PIPE = _loaded                         
    BUNDLE = None                          

#Skapa Flask-app
app = Flask(__name__)

#Hämta lista över feature-namn ----
def get_feature_names() -> List[str]:
    """
    Returnera vilka features API:t förväntar sig.
    - bundle: returnerar 'feature_order'
    - pipeline: försöker extrahera expanderade namn från ColumnTransformer/OneHotEncoder
    """
    if MODE == "bundle":
        return BUNDLE.get("feature_order", [])

    # Plocka fram rå/expanderade feature-namn för info
    try:
        pre = PIPE.named_steps["preprocess"]     
    except Exception:
        return []                                

    # Numeriska kolumner 
    try:
        num_cols = pre.transformers_[0][2] if pre.transformers_ else []
    except Exception:
        num_cols = []

    # Kategoriska 
    try:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_raw = pre.transformers_[1][2]
        cat_expanded = list(ohe.get_feature_names_out(cat_raw))
    except Exception:
        cat_expanded = []

    return list(num_cols) + cat_expanded

# Gör prediktion från JSON 
def predict_from_payload(payload: Any) -> Dict[str, Any]:
    """
    Tar emot:
      - dict (en rad) eller list[dict] (flera rader)
    Returnerar:
      - dict med {"n": antal, "predictions": [float,...]}
    """
    # Gör om JSON till DataFrame. dict 
    df_in = pd.DataFrame([payload]) if isinstance(payload, dict) else pd.DataFrame(payload)

    if MODE == "bundle":
      
        order = BUNDLE.get("feature_order", list(df_in.columns))
        df_in = df_in.reindex(columns=order, fill_value=0)
        # Skala med samma scaler som vid träning
        X = BUNDLE["scaler"].transform(df_in)
        # Prediktion med modellen
        preds = BUNDLE["model"].predict(X)
        return {"n": len(preds), "predictions": [float(p) for p in preds]}

    else:
        
        preds = PIPE.predict(df_in)
        return {"n": len(preds), "predictions": [float(p) for p in preds]}

# Endpoints 

@app.get("/health")
def health():
    """
    Enkel hälsokontroll. Användbar för att se att API:t kör.
    Returnerar även vilket läge vi är i (bundle/pipeline).
    """
    return {"status": "ok", "mode": MODE}

@app.get("/model_info")
def model_info():
    """
    Visar vilka features som förväntas vid /predict.
    - bundle: visa exakt feature_order (det är de transformerade/ordnade features modellen tränades på).
    - pipeline: visar expanderade features (info), men klienten kan skicka råkolumner så sköter pipelinen resten.
    """
    feats = get_feature_names()
    note = "Bundle: Skicka numeriska/enkodade features i denna ordning." if MODE == "bundle" \
        else "Pipeline: Skicka råkolumner (t.ex. 'diet_quality': 'Good'); preprocess sker i pipelinen."
    return {"mode": MODE, "features": feats, "note": note}

@app.post("/predict")
def predict():
    """
    Tar JSON-body (dict eller lista av dict).
    Returnerar prediktioner i JSON.
    Exempel (bundle, transformerade features):
      {"study_time":4.0,"sleep_hours":7,"attendance_percentage":92,"stress_level":3,
       "diet_quality_e":2,"parental_education_level_e":1,"internet_quality_e":2}
    Exempel (pipeline, råkolumner):
      {"study_time":4.0,"sleep_hours":7,"attendance_percentage":92,"stress_level":3,
       "diet_quality":"Good","parental_education_level":"Bachelor","internet_quality":"Good"}
    """
    payload = request.get_json(force=True)   
    try:
        result = predict_from_payload(payload)  # Gör prediktion
        return jsonify(result)                  
    except Exception as e:
        # Returnera feltext och 400 Bad Request
        return jsonify({"error": str(e)}), 400

#(port 5000) 
if __name__ == "__main__":
    # ger auto-reload och bättre felmeddelanden vid utveckling
    app.run(debug=True)
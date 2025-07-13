from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import uvicorn
import re
import json

app = FastAPI()

# Load the trained SVR model
try:
    with open(r"E:\NLP_Housing\svr_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ SVR model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Load the LLM for parsing natural language
try:
    nlp_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    print("✅ LLM loaded.")
except Exception as e:
    print(f"❌ Failed to load LLM: {e}")
    nlp_pipeline = None

# Binary features
binary_cols = [
    'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'
]

# Full feature list expected by the model
input_features = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]

model_features = input_features + [
    'log_area', 'price_per_sqft', 'area_sq', 'area_per_bed', 'bath_per_bed',
    'parking_per_bed', 'amenity_count', 'area_pref', 'area_ac'
]

# Request model
class Query(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/predict_price/")
def predict_price(q: Query):
    if model is None or nlp_pipeline is None:
        raise HTTPException(status_code=500, detail="Model or LLM not available")

    try:
        text = q.query

        # Prompt to LLM
        prompt = (
            "You are a helpful assistant. From the user's input, extract the following features "
            "and return a valid JSON object with only these keys:\n\n"
            "- area (in square feet, integer)\n"
            "- bedrooms (integer)\n"
            "- bathrooms (integer)\n"
            "- stories (integer)\n"
            "- parking (integer)\n"
            "- mainroad (yes/no)\n"
            "- guestroom (yes/no)\n"
            "- basement (yes/no)\n"
            "- hotwaterheating (yes/no)\n"
            "- airconditioning (yes/no)\n"
            "- prefarea (yes/no)\n"
            "- furnishingstatus (choose one: furnished / semi-furnished / unfurnished)\n\n"
            "Return ONLY a valid JSON object.\n\n"
            "Example Input: I have a 3-bedroom house with 2 bathrooms, 8000 square feet, "
            "2 stories, 2-car parking, air conditioning, no basement, yes mainroad access, "
            "guestroom available, in a preferred area and it's semi-furnished.\n\n"
            "Expected Output:\n"
            "{\n"
            "  \"area\": 8000,\n"
            "  \"bedrooms\": 3,\n"
            "  \"bathrooms\": 2,\n"
            "  \"stories\": 2,\n"
            "  \"parking\": 2,\n"
            "  \"mainroad\": \"yes\",\n"
            "  \"guestroom\": \"yes\",\n"
            "  \"basement\": \"no\",\n"
            "  \"hotwaterheating\": \"no\",\n"
            "  \"airconditioning\": \"yes\",\n"
            "  \"prefarea\": \"yes\",\n"
            "  \"furnishingstatus\": \"semi-furnished\"\n"
            "}\n\n"
            f"Now extract this info from the user's message and return the JSON only:\n{text}"
        )
        response = nlp_pipeline(prompt, max_new_tokens=128)[0]['generated_text']
        print("\n🔍 LLM response:", response)

        # Parse and clean response
        info = parse_llm_output(response)
        print("Parsed input:", info)

        # Convert to DataFrame and do feature engineering
        df = pd.DataFrame([info])
        df["log_area"] = np.log1p(df["area"])
        df["price_per_sqft"] = info["area"] / max(info["area"], 1)
        df["area_sq"] = df["area"] ** 2
        df["area_per_bed"] = df["area"] / df["bedrooms"].replace(0, 1)
        df["bath_per_bed"] = df["bathrooms"] / df["bedrooms"].replace(0, 1)
        df["parking_per_bed"] = df["parking"] / (df["bedrooms"] + 1)
        df["amenity_count"] = df[binary_cols].sum(axis=1)
        df["area_pref"] = df["area"] * df["prefarea"]
        df["area_ac"] = df["area"] * df["airconditioning"]

        df = df.reindex(columns=model_features, fill_value=0)
        y_log_pred = model.predict(df)
        y_pred = np.exp(y_log_pred)

        return {
            "predicted_price": float(y_pred[0]),
            "parsed_features": info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def parse_llm_output(text):
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        json_text = text[json_start:json_end]
        parsed = json.loads(json_text)
        parsed = {k.lower().strip(): v for k, v in parsed.items()}

        d = {k: 0 for k in input_features}
        d['area'] = int(parsed.get('area', 7500))
        d['bedrooms'] = int(parsed.get('bedrooms', 3))
        d['bathrooms'] = int(parsed.get('bathrooms', 2))
        d['stories'] = int(parsed.get('stories', 2))
        d['parking'] = int(parsed.get('parking', 1))

        for col in binary_cols:
            d[col] = 1 if str(parsed.get(col, "no")).lower() in ['yes', '1', 'true'] else 0

        furn = str(parsed.get('furnishingstatus', '')).lower()
        if "semi" in furn:
            d['furnishingstatus_semi-furnished'] = 1
        elif "unfurnished" in furn:
            d['furnishingstatus_unfurnished'] = 1

        return d
    except Exception as e:
        print(f"⚠️ Failed to parse LLM response. Fallback. Error: {e}")
        return {
            'area': 7500, 'bedrooms': 3, 'bathrooms': 2, 'stories': 2,
            'mainroad': 1, 'guestroom': 0, 'basement': 0, 'hotwaterheating': 0,
            'airconditioning': 1, 'parking': 1, 'prefarea': 0,
            'furnishingstatus_semi-furnished': 0,
            'furnishingstatus_unfurnished': 0
        }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

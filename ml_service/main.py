from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference_engine import LungCancerExpertSystem


# 1. Initialize the FastAPI web server
app = FastAPI(title="Mediguide AI Engine", version="1.0.0", description="Hybrid AI Expert System")

# 2. Add CORS Middleware (Allow frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # During development, we allow any port to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Boot up the Knowledge Base / ML Engine ONCE in global memory
print("Loading Mediguide Expert Systems int RAM...")
lung_cancer_engine = LungCancerExpertSystem()
print("Systems Ready.")

# 4. Define the strict JSON Schema (The Bouncer)
class PatientRequest(BaseModel):
    symptoms: dict


# 5. Define the web endpoint
@app.post("/predict/lung_cancer")
def predict_lung_cancer(request: PatientRequest):

    """
    Recieves patient Symptoms, processes them through the Expert System, and returns clinical risk, XAI drivers, and rule-based recommendations
    """

    try:
        # hand the raw dictionary off to the specialized engine
        engine_response = lung_cancer_engine.evaluate_patient(request.symptoms)

        # Package the final response to match our CONTRACT.md blueprint
        return {
            "meta": {
                "diseaseType": "Lung Cancer",
                "status": "success"
            },
            "result": engine_response
        }
    except Exception as e:
        # If the ML engine throws a math or column error, catch in gracefully
        raise HTTPException(status_code=500, detail=f"Interface Engine Error: {str(e)}")


    
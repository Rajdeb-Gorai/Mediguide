import os
import joblib
import pandas as pd

class LungCancerExpertSystem:
    def __init__(self):
        """
        FUNDA: Boot up the ML engine. We load the .joblib file into RAM exactly ONCE when the server starts, making API requests lightning-fast.
        """

        # Dynamically find the path to the models/ folder
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, 'models', 'lung_cancer_rf_v1.joblib')

        # Load the Random Forest object
        self.model= joblib.load(model_path)

        # The model memorized the exact column list during training. We save it here.
        self.expected_features = self.model.feature_names_in_

    def evaluate_patient(self, patient_data: dict) -> dict:
        """
        INPUTS: A dictionary of patient symptoms (validated by main.py)
        OUTPUTS: A dictionary containing prediction, probability, confidence, XAI, and clinical rules
        """
        # 1. Feature Alignment (Preventing the Order Bug)
        # Convert dict to a 1-row DataFrame
        input_df = pd.DataFrame([patient_data])

        # Force the exact column order the model expects. Fill missing columns with 0.
        for col in self.expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[self.expected_features]

        # 2. Machine Learning Perception Layer
        prediction = int(self.model.predict(input_df)[0])
        probability = float(self.model.predict_proba(input_df)[0][1]) # Risk of class 1 (Cancer)

        # 3. Confidence Math (Tree Consensus)
        # 0.5 prob = 0% consensus, 1.0 or 0.0 prob = 100% consensus
        confidence = abs(probability-0.5)*2

        # 4. Explanable AI (XAI) Extraction
        importances = self.model.feature_importances_
        # Zip features with their weights, sort highest to lowest
        feature_impacts = sorted(zip(self.expected_features, importances), key=lambda x: x[1], reverse=True)
        top_drivers = [{"feature": f, "weight": round(2, 4)} for f, w in feature_impacts[:3]] # Grab Top 3

        # 5. Expert System Reasoning Layer (The IF-THEN Rules)
        clinical_recommendation = "Standard preventative care."

        if prediction == 1:
            primary_driver = top_drivers[0]["feature"]
            if primary_driver == "ALCOHOL_CONSUMING":
                clinical_recommendation = "URGENT: High risk detected driven by alcohol consumption. Schedule oncology referral and substance cessation counseling."
            elif primary_driver == "SMOKING":
                clinical_recommendation = "URGENT: High risk detected driven by smoking. Schedule oncology referral and nicotine cessation protocol."
            else:
                clinical_recommendation = f"URGENT: High risk detected. Primary contributing factor: {primary_driver}. Schedule immediate consultation."
        
        elif confidence < 0.4: # If trees are split near 50/50
            clinical_recommendation = "BORDERLINE: The AI trees are highly conflicted. Recommend clinical re-evaluation and physical screening to override AI."


        # 6. Bundle the payload for the API
        return {
            "prediction": prediction,
            "probability": round(probability, 3),
            "confidence": round(confidence, 3),
            "xai_drivers": top_drivers,
            "expert_rule_applied": clinical_recommendation
        }



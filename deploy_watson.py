"""
deploy_watson.py
-----------------
Deploys the trained model (models/best_model.pkl) to IBM Watson Machine
Learning so it can be called as a scalable, real-time REST API.

BEFORE RUNNING:
1. Create an IBM Cloud account: https://cloud.ibm.com
2. Create a "Watson Machine Learning" service instance.
3. Create an IBM Cloud API key: Manage > Access (IAM) > API keys.
4. Create a Watson Studio project and note its Space ID (Deployment Spaces).
5. Fill in the four variables below.

Run with:
    python deploy_watson.py
"""

import joblib
from ibm_watson_machine_learning import APIClient

# ---- 1. Fill these in with your own IBM Cloud credentials ----
WML_API_KEY = "YOUR_IBM_CLOUD_API_KEY"
WML_URL = "https://us-south.ml.cloud.ibm.com"   # match your region
SPACE_ID = "YOUR_DEPLOYMENT_SPACE_ID"

wml_credentials = {
    "apikey": WML_API_KEY,
    "url": WML_URL
}

client = APIClient(wml_credentials)
client.set.default_space(SPACE_ID)

# ---- 2. Load the locally trained model ----
model = joblib.load('models/best_model.pkl')

# ---- 3. Find the correct scikit-learn software spec ----
software_spec_uid = client.software_specifications.get_uid_by_name('runtime-23.1-py3.10')

# ---- 4. Store (upload) the model to Watson Machine Learning ----
metadata = {
    client.repository.ModelMetaNames.NAME: 'Credit Card Approval Model',
    client.repository.ModelMetaNames.TYPE: 'scikit-learn_1.1',
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid
}

published_model = client.repository.store_model(
    model=model,
    meta_props=metadata
)
model_uid = client.repository.get_model_id(published_model)
print("Model stored in Watson ML with ID:", model_uid)

# ---- 5. Deploy the model as an online scoring endpoint ----
deployment_metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "Credit Card Approval Deployment",
    client.deployments.ConfigurationMetaNames.ONLINE: {}
}

deployment = client.deployments.create(model_uid, meta_props=deployment_metadata)
deployment_uid = client.deployments.get_id(deployment)
print("Deployed! Deployment ID:", deployment_uid)

scoring_endpoint = client.deployments.get_scoring_href(deployment)
print("Scoring endpoint:", scoring_endpoint)

# ---- Example: how to call the deployed model afterward ----
# payload = {
#     "input_data": [{
#         "fields": feature_cols,   # same order as models/feature_cols.pkl
#         "values": [[1, 1, 1, 0, 75000, 4, 1, 1, 1, 35, 5, 1, 1, 1, 3, 0]]
#     }]
# }
# result = client.deployments.score(deployment_uid, payload)
# print(result)

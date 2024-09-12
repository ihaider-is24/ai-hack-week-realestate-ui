from fastapi import FastAPI, File, UploadFile
import boto3
import io
from PIL import Image
import numpy as np
import json
from mangum import Mangum

app = FastAPI()

handler = Mangum(app)

# Initialize SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='eu-west-1')

# Your SageMaker endpoint name
SAGEMAKER_ENDPOINT = "realestate-classification-final-dataset-v1-endpoint"


def query_endpoint(img):
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType='application/x-image',
        Body=img,
        Accept='application/json;verbose'
    )
    # print(response['Body'].read())
    return response
    

def parse_prediction(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    predicted_label = model_predictions['predicted_label']
    labels = model_predictions['labels']
    probabilities = model_predictions['probabilities']
    return predicted_label, probabilities, labels 


def preprocess_image(image_file):
    """Preprocess the image as per model's requirement."""
    image = Image.open(image_file)
    # Resize or process the image as required by your model
    image = image.resize((224, 224))  # Example resizing to 224x224
    img_array = np.array(image) / 255.0  # Normalize the image if needed
    return img_array

@app.get("/")
def test_endpoint():
    return {"output": "hello world!"}
    

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Read the image content
    img = await file.read()

    try:
        # Call the SageMaker endpoint
        respose = query_endpoint(img)

        # Get the response from the SageMaker endpoint
        result = parse_prediction(respose)

        # Assuming result is in JSON format
        return {"response": result}

    except Exception as e:
        return {"error": str(e)}



# cors
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domain(s) if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# python3 -m venv create_layer
# source create_layer/bin/activate
# pip install -r requirements.txt

# mkdir python
# cp -r create_layer/lib python/
# zip -r layer_content.zip python
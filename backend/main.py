from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

# Project root directory (one level up from backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import the inference logic
from backend.inference import run_deepfake_inference

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# Setup CORS to allow Next.js frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to your frontend domain (e.g., http://localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Detection API."}

@app.post("/api/detect")
async def detect_deepfake(video: UploadFile = File(...)):
    if not video.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an MP4, AVI, or MOV.")

    # Create a unique temporary file path
    temp_filename = f"{uuid.uuid4()}_{video.filename}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        # Save the uploaded file locally so OpenCV can read it
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        # Run the massive ML pipeline
        probability, input_tensor = run_deepfake_inference(temp_path)
        
        is_fake = probability >= 0.5
        
        # Round probability for a cleaner frontend display
        confidence = round(probability * 100, 2) if is_fake else round((1 - probability) * 100, 2)
        
        # Generate GradCAM Heatmap Base64 (Heavy, loaded gracefully)
        try:
            from backend.inference import generate_gradcam_base64
            gradcam_base64 = generate_gradcam_base64(input_tensor)
        except Exception as e:
            print(f"Error generating GradCAM: {e}")
            gradcam_base64 = None
            
        return {
            "success": True,
            "filename": video.filename,
            "is_fake": is_fake,
            "fake_probability": probability,
            "confidence_percentage": f"{confidence}%",
            "gradcam_base64": gradcam_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
    finally:
        # Clean up the temporary file silently
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

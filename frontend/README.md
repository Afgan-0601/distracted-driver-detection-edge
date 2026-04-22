# Frontend - Driver Detection UI

A React web UI to test your distracted driver detection API endpoints.

## Setup

```bash
cd frontend
npm install
npm run dev
```

The UI will start at `http://localhost:3000` and proxies API calls to `http://localhost:8000`.

## Features

- 📁 Image upload with preview
- 🎯 Real-time classification + confidence
- ⚠️ Risk assessment with color-coded levels
- 📊 Confidence breakdown across all 10 classes
- 📝 Raw JSON response viewer

## Requirements

- Make sure your API is running: `uvicorn api.main:app --reload`
- Node.js 16+ installed

## Usage

1. Click **Upload Image** to select a driver photo
2. Click **Analyze Image** to send to the API
3. View the classification, risk level, and detailed scores
4. Click **Reset** to analyze another image

## Adding Example Images

To add example images to the gallery:

1. Place image files in `frontend/public/samples/` — e.g.:
   - `img_c0_1.jpg`
   - `img_c1_1.jpg`
   - `img_c9_1.jpg`

2. Update `frontend/public/samples/manifest.json`:
   ```json
   [
     {
       "id": 1,
       "label": "Safe Driving",
       "path": "/samples/img_c0_1.jpg",
       "thumbnail": "/samples/img_c0_1.jpg"
     },
     {
       "id": 2,
       "label": "Texting",
       "path": "/samples/img_c1_1.jpg",
       "thumbnail": "/samples/img_c1_1.jpg"
     }
   ]
   ```

3. Refresh the browser — examples will appear below the title

**Tip:** Copy sample images from `data/raw/imgs/train/c0/`, `c1/`, `c9/`, etc.


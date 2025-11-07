# Backend Troubleshooting Guide

## CORS Error: "CORS request did not succeed" with status code (null)

This error usually means **the backend server is not running** or not accessible.

## Quick Diagnosis Steps

### 1. Check if backend is running

Open a terminal and run:
```bash
curl http://localhost:8000/health
```

Or use the test script:
```bash
cd backend
python test_backend.py
```

**Expected output if running:**
```json
{"status":"ok"}
```

**If you get connection errors**, the backend is not running.

### 2. Start the backend server

```bash
cd backend
python -m app.main
```

**What to look for in the logs:**
- ✓ Model loaded successfully!
- ✓ Device: cpu (or cuda)
- ✓ Threshold: 0.9249592423439026
- Uvicorn running on http://0.0.0.0:8000

**If you see errors**, check:

1. **Model loading error:**
   - Verify `final_model/` directory exists in project root
   - Check all files are present (config.json, model.safetensors, tokenizer files)
   - Verify `threshold.json` exists in project root

2. **Import errors:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Port already in use:**
   - Check if port 8000 is already in use
   - Change port: `uvicorn app.main:app --port 8001`

### 3. Verify CORS configuration

The backend allows these origins by default:
- http://localhost:5173 (Vite default)
- http://localhost:3000 (React default)
- http://localhost:5174
- http://127.0.0.1:5173
- http://127.0.0.1:3000
- http://127.0.0.1:5174

If your frontend runs on a different port, set environment variable:
```bash
export CORS_ORIGINS="http://localhost:YOUR_PORT"
python -m app.main
```

### 4. Check firewall/antivirus

Some firewalls or antivirus software block localhost connections.
- Temporarily disable to test
- Add exception for localhost:8000

### 5. Browser console errors

Check browser console (F12) for:
- Network tab: See if request reaches server
- Console tab: Check for JavaScript errors

## Common Issues and Solutions

### Issue: "Failed to initialize model"
**Solution:**
- Check `final_model/` directory exists and has all files
- Verify file permissions
- Check disk space

### Issue: "Port 8000 already in use"
**Solution:**
```bash
# Find process using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Kill process or use different port
uvicorn app.main:app --port 8001
```

### Issue: "ModuleNotFoundError"
**Solution:**
```bash
cd backend
pip install -r requirements.txt
```

### Issue: Backend starts but frontend can't connect
**Solution:**
1. Verify backend URL in frontend matches backend port
2. Check CORS origins include frontend URL
3. Try accessing backend directly: `http://localhost:8000/docs`

## Testing the Backend

Once backend is running, test with:

```bash
# Health check
curl http://localhost:8000/health

# Moderate a comment
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test comment"}'
```

Or visit: `http://localhost:8000/docs` for interactive API documentation.


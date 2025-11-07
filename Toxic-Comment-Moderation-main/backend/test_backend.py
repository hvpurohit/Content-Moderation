#!/usr/bin/env python3
"""
Quick test script to verify backend is running and accessible.
Run this before starting the frontend to diagnose connection issues.
"""
import requests
import sys

BACKEND_URL = "http://localhost:8000"

def test_backend():
    print("Testing backend connection...")
    print(f"Backend URL: {BACKEND_URL}")
    print("-" * 60)

    try:
        # Test health endpoint
        print("1. Testing /health endpoint...")
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ Health check passed: {response.json()}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("   ✗ Connection failed - Backend is not running!")
        print("\nTo start the backend:")
        print("   cd backend")
        print("   python -m app.main")
        print("\nOr using uvicorn:")
        print("   cd backend")
        print("   uvicorn app.main:app --reload")
        return False

    except requests.exceptions.Timeout:
        print("   ✗ Request timed out - Backend is not responding")
        return False

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    try:
        # Test moderate endpoint
        print("\n2. Testing /moderate endpoint...")
        response = requests.post(
            f"{BACKEND_URL}/moderate",
            json={"text": "test"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            print(f"   ✓ Moderate endpoint works: {response.json()}")
        else:
            print(f"   ✗ Moderate endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"   ✗ Error testing moderate endpoint: {e}")
        return False

    print("\n" + "-" * 60)
    print("✓ All tests passed! Backend is running correctly.")
    return True

if __name__ == "__main__":
    success = test_backend()
    sys.exit(0 if success else 1)


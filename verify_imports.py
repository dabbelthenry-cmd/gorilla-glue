import sys
import os

# Add the directory to the path
sys.path.append('/Users/kishansingh/Documents/Kishan/Gorilla Glue ')

try:
    import data
    import ML_engine
    import suggestions
    import database
    # import app # app.py runs streamlit stuff which might fail in headless script without proper setup, but let's try importing it to check syntax
    print("All modules imported successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

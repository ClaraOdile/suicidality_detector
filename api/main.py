import threading
import uvicorn
import streamlit as st
import fast
import app

# Run FastAPI app in a separate thread
def run_fastapi():
    uvicorn.run(fast, host='0.0.0.0', port=8000)

# Run Streamlit app in a separate thread
def run_streamlit():
    app.run()

if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi)
    streamlit_thread = threading.Thread(target=run_streamlit)

    fastapi_thread.start()
    streamlit_thread.start()

    fastapi_thread.join()
    streamlit_thread.join()

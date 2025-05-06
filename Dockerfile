FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add gradio for Hugging Face Spaces interface
RUN pip install gradio

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p ./temp ./plots ./reports ./data

# Expose the port the app runs on (Spaces uses port 7860)
EXPOSE 7860

# Set environment variables
ENV PORT=7860

# Command to run the application
CMD ["python", "huggingface_app.py"] 
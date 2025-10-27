# Use official Python 3.13 image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (helps Docker cache dependencies)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code
COPY . .

# Create artifacts folder to store trained model
RUN mkdir -p artifacts

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

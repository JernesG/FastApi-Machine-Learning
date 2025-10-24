# 1 Use official Python 3.13 image
FROM python:3.13-slim

# 2 Set the working directory inside the container
WORKDIR /app

# 3 Copy requirements first (helps Docker cache dependencies)
COPY requirements.txt .

# 4 Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5 Copy the rest of your project code
COPY . .

# 6 Create artifacts folder to store trained model
RUN mkdir -p artifacts

# 7 Expose FastAPI port
EXPOSE 8000

# 8 Start the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

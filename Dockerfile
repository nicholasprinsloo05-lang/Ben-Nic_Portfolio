# 1. Start from an official Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy remaining application files
COPY . .

# 6. Expose application port
EXPOSE 8501

# 7. Start the application
CMD ["streamlit", "run", "app_regression.py"]
CMD ["streamlit", "run", "Home.py"]
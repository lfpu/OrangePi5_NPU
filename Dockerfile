# Use an official Python runtime as a parent image
FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0


# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
COPY startRKNN.sh  /startRKNN.sh
# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
ENTRYPOINT [ "/startRKNN.sh" ]


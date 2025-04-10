FROM python:3.10-slim

# Install bash just in case
#RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
RUN pip install google-cloud-storage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY datasets/tgbl_review /app/datasets/tgbl_review
#COPY .tgb_cache /app/.tgb_cache

# Copy experiment scripts
COPY . .

# Give execute permissions to your bash script
RUN chmod +x experiments.sh

# Run the bash script
CMD ["./experiments.sh"]


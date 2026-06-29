FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

# TA-Lib
COPY --from=ta-lib/ta-lib:latest /usr/lib/libta_lib* /usr/lib/
COPY --from=ta-lib/ta-lib:latest /usr/include/ta-lib/ /usr/include/ta-lib/

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

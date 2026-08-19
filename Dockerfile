FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY smc_regime/ smc_regime/

ENTRYPOINT ["python", "-m", "smc_regime.daily_snapshot"]
CMD ["--intervals", "1d,1h"]

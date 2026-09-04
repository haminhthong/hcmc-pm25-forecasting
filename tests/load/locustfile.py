from locust import HttpUser, between, task

SAMPLE_PAYLOAD = [
    {
        "timestamp": f"2024-01-01T{h:02d}:00:00",
        "station": "Tram_A",
        "PM2.5": 20.0 + (h % 5),
        "O3": 10.0,
        "SO2": 5.0,
    }
    for h in range(25)
]


class PM25User(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict(self):
        self.client.post(
            "/predict",
            json={"observations": SAMPLE_PAYLOAD},
        )

    @task(1)
    def health_check(self):
        self.client.get("/health")

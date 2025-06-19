from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrik untuk pemantauan API
TOTAL_REQ = Counter('api_total_requests', 'Jumlah total permintaan API')
REQ_LATENCY = Histogram('api_response_duration_seconds', 'Durasi respon API dalam detik')
REQ_FAILED = Counter('api_error_count', 'Jumlah total error')
REQ_SUCCESS = Counter('api_success_count', 'Jumlah total permintaan sukses')
INCOMING_SIZE = Histogram('api_input_payload_bytes', 'Ukuran data permintaan (bytes)')
OUTGOING_SIZE = Histogram('api_output_payload_bytes', 'Ukuran data balasan (bytes)')

# Metrik untuk pemantauan sistem
METRIC_CPU = Gauge('monitor_cpu_usage_pct', 'Persentase penggunaan CPU')
METRIC_RAM = Gauge('monitor_ram_usage_pct', 'Persentase penggunaan RAM')
SWAP_USAGE = Gauge('monitor_swap_usage_pct', 'Persentase penggunaan Swap Memory')  # Metrik baru
NET_BYTES_SENT = Gauge('monitor_network_bytes_sent', 'Total byte jaringan yang dikirim')
NET_BYTES_RECV = Gauge('monitor_network_bytes_received', 'Total byte jaringan yang diterima')

# Endpoint untuk halaman utama
@app.route('/')
def index():
    return "Prometheus Exporter is running. Access /metrics for metrics."

# Endpoint untuk Prometheus
@app.route('/metrics', methods=['GET'])
def export_metrics():
    METRIC_CPU.set(psutil.cpu_percent(interval=1))
    METRIC_RAM.set(psutil.virtual_memory().percent)
    SWAP_USAGE.set(psutil.swap_memory().percent)  # Gunakan swap memory
    net_data = psutil.net_io_counters()
    NET_BYTES_SENT.set(net_data.bytes_sent)
    NET_BYTES_RECV.set(net_data.bytes_recv)

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Endpoint utama untuk melakukan prediksi dan mencatat metrik
@app.route('/predict', methods=['POST'])
def call_model():
    timestamp_start = time.time()
    TOTAL_REQ.inc()

    model_endpoint = "http://127.0.0.1:5005/invocations"
    input_data = request.get_json()

    try:
        res = requests.post(model_endpoint, json=input_data)
        response_time = time.time() - timestamp_start
        REQ_LATENCY.observe(response_time)

        INCOMING_SIZE.observe(len(request.data))
        OUTGOING_SIZE.observe(len(res.content))

        if res.status_code == 200:
            REQ_SUCCESS.inc()
        else:
            REQ_FAILED.inc()

        return jsonify(res.json())

    except Exception as err:
        REQ_FAILED.inc()
        return jsonify({"error": str(err)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)

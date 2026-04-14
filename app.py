"""
AI-Driven Logistics and Route Optimization System
Flask Backend
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import random
import math
import heapq
from collections import defaultdict
import time
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

CITIES = {
    "Mumbai":     {"lat": 19.076, "lng": 72.877, "zone": "west"},
    "Pune":       {"lat": 18.520, "lng": 73.856, "zone": "west"},
    "Nashik":     {"lat": 19.998, "lng": 73.789, "zone": "west"},
    "Surat":      {"lat": 21.170, "lng": 72.831, "zone": "west"},
    "Ahmedabad":  {"lat": 23.023, "lng": 72.572, "zone": "west"},
    "Delhi":      {"lat": 28.613, "lng": 77.209, "zone": "north"},
    "Jaipur":     {"lat": 26.912, "lng": 75.787, "zone": "north"},
    "Lucknow":    {"lat": 26.847, "lng": 80.947, "zone": "north"},
    "Agra":       {"lat": 27.177, "lng": 78.007, "zone": "north"},
    "Chandigarh": {"lat": 30.733, "lng": 76.779, "zone": "north"},
    "Bangalore":  {"lat": 12.972, "lng": 77.594, "zone": "south"},
    "Chennai":    {"lat": 13.083, "lng": 80.270, "zone": "south"},
    "Hyderabad":  {"lat": 17.385, "lng": 78.486, "zone": "south"},
    "Kochi":      {"lat":  9.931, "lng": 76.267, "zone": "south"},
    "Coimbatore": {"lat": 11.017, "lng": 76.956, "zone": "south"},
    "Kolkata":    {"lat": 22.572, "lng": 88.363, "zone": "east"},
    "Bhubaneswar":{"lat": 20.296, "lng": 85.825, "zone": "east"},
    "Patna":      {"lat": 25.611, "lng": 85.143, "zone": "east"},
    "Nagpur":     {"lat": 21.145, "lng": 79.082, "zone": "central"},
    "Indore":     {"lat": 22.719, "lng": 75.857, "zone": "central"},
}

CITY_NAMES = list(CITIES.keys())

BASE_EDGES = [
    ("Mumbai",    "Pune",        148),
    ("Mumbai",    "Nashik",      167),
    ("Mumbai",    "Surat",       284),
    ("Mumbai",    "Nagpur",      835),
    ("Pune",      "Nashik",      211),
    ("Pune",      "Hyderabad",   559),
    ("Pune",      "Nagpur",      590),
    ("Surat",     "Ahmedabad",   265),
    ("Ahmedabad", "Jaipur",      670),
    ("Ahmedabad", "Indore",      395),
    ("Delhi",     "Jaipur",      270),
    ("Delhi",     "Agra",        210),
    ("Delhi",     "Chandigarh",  250),
    ("Delhi",     "Lucknow",     500),
    ("Jaipur",    "Agra",        235),
    ("Jaipur",    "Indore",      490),
    ("Lucknow",   "Agra",        340),
    ("Lucknow",   "Patna",       570),
    ("Bangalore", "Chennai",     346),
    ("Bangalore", "Hyderabad",   570),
    ("Bangalore", "Kochi",       530),
    ("Bangalore", "Coimbatore",  360),
    ("Chennai",   "Coimbatore",  495),
    ("Chennai",   "Hyderabad",   625),
    ("Hyderabad", "Nagpur",      505),
    ("Kolkata",   "Bhubaneswar", 440),
    ("Kolkata",   "Patna",       600),
    ("Nagpur",    "Indore",      455),
    ("Indore",    "Mumbai",      590),
]


def haversine(city1, city2):
    c1, c2 = CITIES[city1], CITIES[city2]
    R = 6371
    lat1, lat2 = math.radians(c1["lat"]), math.radians(c2["lat"])
    dlat = math.radians(c2["lat"] - c1["lat"])
    dlng = math.radians(c2["lng"] - c1["lng"])
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_graph(traffic_factor=1.0):
    graph = defaultdict(list)
    for u, v, dist in BASE_EDGES:
        noise = random.uniform(0.92, 1.08) * traffic_factor
        w = round(dist * noise, 1)
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph


def dijkstra(graph, start, end):
    dist = {city: float('inf') for city in CITIES}
    dist[start] = 0
    prev = {}
    pq = [(0, start)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            break
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    path, node = [], end
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(start)
    path.reverse()
    return path, round(dist[end], 1)


def astar(graph, start, end):
    open_set = [(0, start)]
    g = defaultdict(lambda: float('inf'))
    g[start] = 0
    came_from = {}
    visited = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == end:
            break
        for neighbor, weight in graph[current]:
            tentative_g = g[current] + weight
            if tentative_g < g[neighbor]:
                came_from[neighbor] = current
                g[neighbor] = tentative_g
                f = tentative_g + haversine(neighbor, end)
                heapq.heappush(open_set, (f, neighbor))
    path, node = [], end
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path, round(g[end], 1)


def forecast_demand():
    hour = datetime.now().hour
    base_weights = {
        "Mumbai": 95, "Delhi": 92, "Bangalore": 88, "Chennai": 75,
        "Hyderabad": 72, "Kolkata": 70, "Pune": 65, "Ahmedabad": 62,
        "Jaipur": 55, "Surat": 58, "Lucknow": 52, "Nagpur": 48,
        "Indore": 45, "Agra": 42, "Patna": 40, "Bhubaneswar": 38,
        "Nashik": 35, "Kochi": 50, "Coimbatore": 44, "Chandigarh": 48,
    }
    peak = 1.4 if (9 <= hour <= 12 or 17 <= hour <= 20) else 0.85
    result = {}
    for city, base in base_weights.items():
        val = base * peak * random.uniform(0.88, 1.12)
        result[city] = {
            "demand": round(val),
            "trend": random.choice(["up", "up", "stable", "down"]),
            "confidence": round(random.uniform(0.82, 0.97), 2),
            "next_day": round(val * random.uniform(0.95, 1.15)),
        }
    return result


def get_traffic_data():
    hour = datetime.now().hour
    is_peak = 8 <= hour <= 10 or 17 <= hour <= 20
    result = []
    for u, v, dist in BASE_EDGES:
        level = random.choices(
            ["low", "moderate", "high", "severe"],
            weights=[30, 35, 25, 10] if is_peak else [50, 30, 15, 5]
        )[0]
        delay = {"low": 0, "moderate": 15, "high": 35, "severe": 60}[level]
        result.append({
            "from": u, "to": v,
            "distance": dist,
            "traffic": level,
            "delay_pct": delay,
            "effective_dist": round(dist * (1 + delay / 100), 1),
        })
    return result


def get_fleet():
    vehicles = []
    types = [("Truck", 10000, 12), ("Van", 3000, 18), ("Bike", 500, 30)]
    cities = random.sample(CITY_NAMES, 12)
    for i, city in enumerate(cities):
        vtype, cap, speed = random.choice(types)
        load = random.randint(0, cap)
        vehicles.append({
            "id": f"VH-{1001 + i}",
            "type": vtype,
            "location": city,
            "capacity_kg": cap,
            "load_kg": load,
            "load_pct": round(load / cap * 100),
            "speed_kmh": speed,
            "status": random.choice(["active", "active", "active", "idle", "maintenance"]),
            "fuel_pct": random.randint(20, 100),
            "trips_today": random.randint(0, 8),
        })
    return vehicles


def get_kpis():
    return {
        "total_deliveries": random.randint(1200, 1600),
        "on_time_pct": round(random.uniform(87, 97), 1),
        "avg_route_km": round(random.uniform(310, 420), 1),
        "fuel_saved_pct": round(random.uniform(14, 22), 1),
        "cost_saved_inr": random.randint(180000, 320000),
        "active_vehicles": random.randint(28, 45),
        "pending_orders": random.randint(80, 200),
        "alerts": random.randint(2, 8),
    }


def get_historical():
    days = [(datetime.now() - timedelta(days=i)).strftime("%b %d") for i in range(13, -1, -1)]
    return {
        "days": days,
        "deliveries": [random.randint(900, 1500) for _ in days],
        "fuel_efficiency": [round(random.uniform(72, 91), 1) for _ in days],
        "costs": [random.randint(45000, 90000) for _ in days],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard")
def api_dashboard():
    return jsonify({
        "kpis": get_kpis(),
        "fleet": get_fleet(),
        "traffic": get_traffic_data(),
        "demand": forecast_demand(),
        "historical": get_historical(),
        "cities": {k: {"lat": v["lat"], "lng": v["lng"], "zone": v["zone"]} for k, v in CITIES.items()},
        "edges": [{"from": u, "to": v, "dist": d} for u, v, d in BASE_EDGES],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    data = request.json
    origin = data.get("origin", "Mumbai")
    destination = data.get("destination", "Delhi")
    algorithm = data.get("algorithm", "astar")
    use_traffic = data.get("traffic", True)

    if origin not in CITIES or destination not in CITIES:
        return jsonify({"error": "Invalid city"}), 400
    if origin == destination:
        return jsonify({"error": "Origin and destination must be different"}), 400

    tf = random.uniform(1.1, 1.5) if use_traffic else 1.0
    graph = build_graph(tf)

    t0 = time.time()
    if algorithm == "dijkstra":
        path, dist = dijkstra(graph, origin, destination)
        algo_name = "Dijkstra"
    else:
        path, dist = astar(graph, origin, destination)
        algo_name = "A* (AI Heuristic)"
    elapsed = round((time.time() - t0) * 1000, 3)

    # baseline without traffic
    g_base = build_graph(1.0)
    _, base_dist = astar(g_base, origin, destination)
    saving = round(((base_dist - dist) / base_dist * 100) if base_dist > 0 else 0, 1)

    speed = 55
    travel_h = round(dist / speed, 2)
    travel_str = f"{int(travel_h)}h {int((travel_h % 1)*60)}m"

    coords = [{"city": c, "lat": CITIES[c]["lat"], "lng": CITIES[c]["lng"]} for c in path]

    return jsonify({
        "algorithm": algo_name,
        "path": path,
        "coords": coords,
        "total_distance_km": dist,
        "estimated_time": travel_str,
        "computation_ms": elapsed,
        "traffic_applied": use_traffic,
        "savings_pct": saving,
        "stops": len(path),
    })


@app.route("/api/fleet")
def api_fleet():
    return jsonify(get_fleet())


@app.route("/api/demand")
def api_demand():
    return jsonify(forecast_demand())


@app.route("/api/traffic")
def api_traffic():
    return jsonify(get_traffic_data())


if __name__ == "__main__":
    print("=" * 55)
    print("  AI Logistics & Route Optimization System")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)
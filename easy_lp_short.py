"""
EASY PROBLEM (Linear Program) -- short version
------------------------------------------------
Decide, for every EV, how to get to its destination:  drive directly, or charge at a
station (which station, starting when, how fast), so that
    total cost = trip time + waiting + electricity bill + station peak load   is smallest,
without overloading any station's grid and without the battery ever going below 15%.

Run:  python easy_lp_short.py        (the two CSV files must be in the same folder)
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ------------------------------------------------------------------ settings
K, H = 100, 64                 # 100 vehicles, 64 time slots of 15 min (08:00 -> 24:00)
START_HOUR, SLOT_H = 8, 0.25   # slot 0 = 08:00, one slot = 0.25 hour
SOC_MIN, SOC_TARGET = 0.15, 0.80   # battery never below 15%; charging goes up to 80%
ROAD = 1.25                    # road distance = 1.25 x straight line (assumed)
MAX_WAIT = 8                   # charging may start up to 8 slots (2 h) after arrival
LEVELS = [1.0, 0.5, 0.33]      # charge at full, half or one-third power
GRID_SHARE = 0.4               # assumed: grid can feed 40% of the plugs at average rate
W_TIME, W_WAIT, W_ENERGY, W_PEAK, W_FAIL = 1.0, 2.0, 1.0, 0.2, 5000.0   # objective weights
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ 1. read the data
ev = pd.read_csv("ev_charging_dataset.csv")          # vehicles + hourly averages
loc = pd.read_csv("location_dataset.csv")            # the map
xy = loc["Coordinates"].str.strip("()").str.split(",", expand=True).astype(float).to_numpy()
stations = loc.index[loc["Type"] == "Power Plant"].tolist()    # stations sit at the Power Plants
S, st_xy = len(stations), xy[stations]

hour = pd.to_datetime(ev["Date_Time"]).dt.hour
load_by_hour = ev.groupby(hour)["Charging_Load_kW"].mean().to_numpy()    # forecast = hour-of-day average
queue_by_hour = ev.groupby(hour)["Queue_Time_mins"].mean().to_numpy()
SPEED = 55 / (1 + 0.06 * ev["Traffic_Data"].mean())                      # km/h (assumed formula)
FACTOR = (1 + 0.03 * ev["Traffic_Data"].mean()) * (1 + 0.005 * abs(ev["Temperature_C"].mean() - 22))  # extra energy use

# ------------------------------------------------------------------ 2. time slots and stations
clock = START_HOUR + np.arange(H) * SLOT_H                    # clock time of each slot
hour_of_slot = (clock % 24).astype(int)
price = np.where((clock >= 17) & (clock < 21), 14, np.where((clock >= 11) & (clock < 17), 9, 6))  # assumed tariff
busy = rng.uniform(0.8, 1.2, S)[:, None]                      # some stations a bit busier
Lhat = busy * load_by_hour[hour_of_slot]                      # background load forecast (S x H), kW
Qhat = busy * queue_by_hour[hour_of_slot]                     # queue time forecast (S x H), minutes
plugs = np.maximum(rng.choice(ev["Station_Capacity_EV"].to_numpy(), S), 2)
P_STATION = ev["Charging_Rate_kW"].max()                      # fastest charging rate in the data
G = Lhat.mean(axis=1) + GRID_SHARE * plugs * ev["Charging_Rate_kW"].mean()   # grid limit of each station, kW


# ------------------------------------------------------------------ 3. options of one vehicle
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) * ROAD         # km


def option(k, kind, s=-1, t=0, slots=0, power=0.0, time=0.0, wait=0.0, bill=0.0):
    """One possible choice of vehicle k, with its cost already computed."""
    cost = W_FAIL if kind == "fail" else W_TIME * time + W_WAIT * wait + W_ENERGY * bill
    return dict(vehicle=k, kind=kind, station=s, start=t, slots=slots, power=power, cost=cost)


def make_options(k, B, soc0, cons, Pk, origin, dest, dep):
    """All choices of one vehicle.  Any choice that would push the battery below 15% is never created."""
    opts = []
    # --- choice 1: drive directly
    d = dist(xy[origin], xy[dest])
    if soc0 - d * cons * FACTOR / B >= SOC_MIN:
        opts.append(option(k, "direct", time=d / SPEED * 60))
    # --- choice 2: charge at station s, starting in slot t, at some power
    for s in range(S):
        d1 = dist(xy[origin], st_xy[s])                         # start -> station
        soc_arrive = soc0 - d1 * cons * FACTOR / B
        if soc_arrive < SOC_MIN or soc_arrive >= SOC_TARGET:    # can't reach it / no need to charge
            continue
        d2 = dist(st_xy[s], xy[dest])                           # station -> destination
        if SOC_TARGET - d2 * cons * FACTOR / B < SOC_MIN:       # can't reach the destination after charging
            continue
        t1 = d1 / SPEED * 60                                    # minutes to reach the station
        arrive = dep + math.ceil(t1 / 15)                       # slot of arrival
        if arrive >= H:
            continue
        energy = B * (SOC_TARGET - soc_arrive)                  # kWh to buy
        for level in LEVELS:
            slots = math.ceil(energy / (level * min(Pk, P_STATION) * SLOT_H))   # slots needed
            power = energy / (slots * SLOT_H)                   # kW, so that power x time = energy
            for t in range(arrive, min(arrive + MAX_WAIT, H - slots) + 1):      # start charging in slot t
                opts.append(option(k, "station", s, t, slots, power,
                                   time=t1 + d2 / SPEED * 60 + slots * 15,       # driving + charging minutes
                                   wait=Qhat[s, arrive] + (t - arrive) * 15,     # queue + deliberate delay
                                   bill=price[t:t + slots].sum() * power * SLOT_H))
    # --- choice 3: fail (keeps the model always solvable)
    opts.append(option(k, "fail"))
    return opts


# ------------------------------------------------------------------ 4. create the vehicles and all options
options, trips = [], []
for k in range(K):
    while True:                                                 # redraw until the vehicle can be served
        row = rng.integers(0, len(ev))                          # a random real row of the CSV
        origin, dest = rng.integers(0, len(loc)), rng.integers(0, len(loc))
        dep = rng.integers(0, 16) if rng.random() < 0.4 else rng.integers(16, 40)   # departure slot
        if origin == dest:
            continue
        opts = make_options(k, ev["Battery_Capacity_kWh"][row], ev["State_of_Charge_%"][row] / 100,
                            ev["Energy_Consumption_Rate_kWh/km"][row], ev["Charging_Rate_kW"][row], origin, dest, dep)
        if len(opts) > 1:                                       # has at least one real option besides "fail"
            break
    options += opts
    trips.append((origin, dest))
n = len(options)
print(f"{K} vehicles, {S} stations, {n} options in total")

# ------------------------------------------------------------------ 5. the LINEAR PROGRAM
# variables:  x[o] in [0,1] for every option o,   peak[s] >= 0 for every station s
# minimise :  sum_o cost[o] * x[o]  +  W_PEAK * sum_s peak[s]
c = np.array([o["cost"] for o in options] + [W_PEAK] * S)

# (C1) every vehicle gets exactly one option:  sum of x[o] over its options = 1
A_eq = np.zeros((K, n + S))
for i, o in enumerate(options):
    A_eq[o["vehicle"], i] = 1
b_eq = np.ones(K)

# (C3) grid never overloaded:  charging power at (station s, slot tau) <= G[s] - Lhat[s, tau]
A_grid = np.zeros((S * H, n + S))                               # row number = s*H + tau
for i, o in enumerate(options):
    if o["kind"] == "station":
        for tau in range(o["start"], o["start"] + o["slots"]):                                                                                      
            A_grid[o["station"] * H + tau, i] = o["power"]
b_grid = np.maximum(G[:, None] - Lhat, 0).flatten()

# (C4) peak[s] >= Lhat[s, tau] + charging power   ->   power - peak[s] <= -Lhat[s, tau]
A_peak = A_grid.copy()
for s in range(S):
    for tau in range(H):
        A_peak[s * H + tau, n + s] = -1
b_peak = -Lhat.flatten()

result = linprog(c, A_ub=np.vstack([A_grid, A_peak]), b_ub=np.concatenate([b_grid, b_peak]),
                 A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * n + [(0, None)] * S, method="highs")

# ------------------------------------------------------------------ 6. result
print("solver status   :", result.message)
print(f"optimal objective: {result.fun:.1f}")
x = result.x[:n]
kinds = {"direct": 0, "station": 0, "fail": 0}
for k in range(K):                                              # each vehicle's main (largest-share) option
    best = max((i for i, o in enumerate(options) if o["vehicle"] == k), key=lambda i: x[i])
    kinds[options[best]["kind"]] += 1
print("vehicles driving directly / charging at a station / unserved:", kinds["direct"], "/", kinds["station"], "/", kinds["fail"])
print("\nplan for the first 5 vehicles:")
for k in range(5):
    best = max((i for i, o in enumerate(options) if o["vehicle"] == k), key=lambda i: x[i])
    o = options[best]
    where = f"charge at {loc['Location'][stations[o['station']]]} from {clock[o['start']]:.2f}h at {o['power']:.0f} kW" if o["kind"] == "station" else o["kind"]
    print(f"  vehicle {k}: {loc['Location'][trips[k][0]]} -> {loc['Location'][trips[k][1]]}: {where}")

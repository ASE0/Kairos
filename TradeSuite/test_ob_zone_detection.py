import pandas as pd
from core.order_block_gate import OrderBlockGate

# Path to the dataset
DATA_PATH = 'workspaces/datasets/NQ_5s_1m.csv'
TARGET_DATE = '2024-03-07'
TAU = 50  # zone lifetime in bars (decay)


def zone_is_expired(zone, bar, bar_index):
    # Expire if too old
    if bar_index - zone['bar_index'] >= TAU:
        return True
    # Expire if price closes through the zone
    close = bar['close']
    if zone['zone_direction'] == 'bearish' and close > zone['zone_max']:
        return True
    if zone['zone_direction'] == 'bullish' and close < zone['zone_min']:
        return True
    return False

def zones_overlap(z1, z2):
    # Returns True if two zones overlap in price
    return not (z1['zone_max'] < z2['zone_min'] or z2['zone_max'] < z1['zone_min'])

def main():
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
    # Filter for the target date
    df_day = df[df['datetime'].str.startswith(TARGET_DATE)].reset_index(drop=True)
    print(f"Loaded {len(df_day)} bars for {TARGET_DATE}")
    # Convert to list of dicts for OB detection
    bars = df_day.to_dict(orient='records')
    # Detect all possible OBs (raw)
    all_zones = OrderBlockGate.detect_zones(
        bars,
        buffer_pts=2,
        gamma=2.0,
        delta=1.5,
        epsilon=1e-4,
        tau=TAU
    )
    print(f"Total OBs detected (raw, no expiry/skip): {len(all_zones)}")
    # Now simulate real-time zone lifecycle
    active_zones = []
    all_active_zones = []  # for reporting
    for i, bar in enumerate(bars):
        # Expire old/invalidated zones
        active_zones = [z for z in active_zones if not zone_is_expired(z, bar, i)]
        # Find new OBs at this bar
        new_zones = [z for z in all_zones if z['bar_index'] == i]
        for z in new_zones:
            # Only add if no overlap with any active zone
            if not any(zones_overlap(z, az) for az in active_zones):
                active_zones.append(z)
        # Record snapshot of active zones at this bar
        all_active_zones.append(list(active_zones))
    # Flatten all_active_zones to unique zones
    unique_active_zones = {z['bar_index']: z for zones in all_active_zones for z in zones}.values()
    print(f"OBs that would be active at any time (GUI-realistic): {len(unique_active_zones)}")
    for idx, z in enumerate(unique_active_zones):
        print(f"Active Zone {idx+1}: bar_index={z['bar_index']}, direction={z['zone_direction']}, min={z['zone_min']}, max={z['zone_max']}")

if __name__ == "__main__":
    main() 
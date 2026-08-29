import os, json

print(f'{"Bearing":<30} {"Condition":>9} {"RMSE":>8} {"PHM Score":>10} {"DeltaRMSE%":>11}')
print('-' * 72)

for f in sorted(os.listdir('results')):
    if f.startswith('pronostia_') and f.endswith('.json'):
        with open(os.path.join('results', f)) as fh:
            d = json.load(fh)
        name = f.replace('pronostia_', '').replace('.json', '')
        print(f"{name:<30} {d['condition_id']:>9} {d['pronostia_rmse']:>8.4f} {d['pronostia_phm2012_score']:>10.4f} {d.get('delta_rmse_drop_pct', 'N/A'):>11}")

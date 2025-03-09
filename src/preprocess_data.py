import pandas as pd

def preprocess_data(energy_file, gas_file, output_file):
    energy = pd.read_csv(energy_file)
    gas = pd.read_csv(gas_file)
    columns = ['time', 'energy_usage']
    df = pd.DataFrame(columns=columns)
    energy['Import T1 kWh diff'] = energy['Import T1 kWh'].diff()
    energy['Import T2 kWh diff'] = energy['Import T2 kWh'].diff()
    energy['Export T1 kWh diff'] = energy['Export T1 kWh'].diff()
    energy['Export T2 kWh diff'] = energy['Export T2 kWh'].diff()

    import_energy = energy['Import T1 kWh diff'].values + energy['Import T2 kWh diff'].values
    export_energy = energy['Export T1 kWh diff'].values + energy['Export T2 kWh diff'].values
    df['time'] = energy['time']
    df['energy_usage'] = export_energy - import_energy
    df.to_csv(output_file, index=False)

preprocess_data('data/energy_usage.csv', 'data/gas_usage.csv', 'data/energy_gas_usage.csv')
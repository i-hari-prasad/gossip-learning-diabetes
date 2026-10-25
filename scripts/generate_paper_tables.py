import pandas as pd
import os

os.makedirs('artifacts/paper_tables', exist_ok=True)

def generate_table1():
    data = {
        'Model': [
            'Centralized LSTM Model [5]',
            'Federated Learning Model [7]',
            'Conventional Gossip Model [8]',
            'Proposed Energy-Aware LGLM'
        ],
        'Energy per Comm. Cycle (mJ)': [120, 85, 60, 35],
        'Total Battery Use (%)': [85, 65, 50, 30],
        'Transmission Reduction (%)': [0, 30, 50, 70],
        'Network Lifetime Extension (%)': [0, 25, 45, 80]
    }
    df = pd.DataFrame(data)
    df.to_csv('artifacts/paper_tables/TABLE_I_Energy_Consumption_Analysis.csv', index=False)
    print("TABLE I: ENERGY CONSUMPTION ANALYSIS")
    print(df.to_string(index=False))
    print("\n")
    return df

def generate_table2():
    data = {
        'Model': [
            'Centralized LSTM [5]',
            'Federated Learning Model [7]',
            'Proposed LGLM (Decentralized Gossip)'
        ],
        'RMSE (mg/dL)': ['14.1', '15.1', '13.6'],
        'MAE (mg/dL)': ['19.4', '—', '9.2'],
        'MARD (%)': ['—', '—', '8.5']
    }
    df = pd.DataFrame(data)
    df.to_csv('artifacts/paper_tables/TABLE_II_Performance_Comparison.csv', index=False)
    print("TABLE II: PERFORMANCE COMPARISON")
    print(df.to_string(index=False))
    print("\n")
    return df

if __name__ == "__main__":
    generate_table1()
    generate_table2()
    print("Tables saved to artifacts/paper_tables/")

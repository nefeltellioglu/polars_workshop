#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:13:23 2024

@author: nefeltellioglu
"""

import pandas as pd
import numpy as np
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import time



@dataclass
class Params:
    no_runs: int  
    pop_size: int 
    inf_duration: float
    exposed_duration: float 
    random_seed: int 
    transmission_rate: float
    time_horizon: float
    time_step: float
    record_transmission: bool
    record_all_new_cases: bool

def run_SEIR_model_pd(p: Params):
    secondary_infections_from_seed_infection_list = []
    exposed_by_seed_df = pd.DataFrame()
    all_exposed_cases = pd.DataFrame()
    rng = np.random.RandomState(p.random_seed)
    ts = np.arange(p.time_step, p.time_horizon, p.time_step)    
    transmission_rate = 1 - ((1 - p.transmission_rate)**(p.time_step))
    
    if p.record_transmission:
        possible_states = pd.DataFrame({
            "state": ["Susceptible", "Exposed", "Infectious", "Recovered"],
            "count":  [0, 0, 0, 0]
        })
        all_records = pd.DataFrame()
        
    for run in range(p.no_runs):
        secondary_infections_from_seed_infection = 0
        
        # Initialize population
        population = pd.DataFrame({
            "id": range(p.pop_size),
            "state": ["Susceptible"] * p.pop_size,
            "s_time_exposed": [0.0] * p.pop_size,
            "s_time_infectious": [0.0] * p.pop_size,
            "s_time_recovery": [0.0] * p.pop_size,
            "exposed_from": [-1] * p.pop_size,
            "pop_size": p.pop_size,
            #"hh_id": run,
            "run_no": run,
        })
        
        
        # Infect one individual in the population (seed infection)
        seed_infection_index = 0  # rng.randint(0, p.pop_size - 1)
        population.loc[seed_infection_index, "state"] = "Infectious"
        population.loc[seed_infection_index, "s_time_recovery"] = rng.exponential(p.inf_duration)
        
        if p.record_transmission:
            cur_records = possible_states.copy()
            cur_records['count'] = population['state'].value_counts().reindex(cur_records['state']).fillna(0).values
            cur_records['t'] = ts[0] - p.time_step
            cur_records['run_no'] = run
            all_records = pd.concat([all_records, cur_records], ignore_index=True)
        
        # Simulate transmission in the population
        for t in ts:
            infected_ids = population[population["state"] == "Infectious"]["id"].tolist()
            susceptible_individuals = population[population["state"] == "Susceptible"]
            will_infected = rng.rand(len(susceptible_individuals)) < (transmission_rate * len(infected_ids))
            will_infected_individuals = susceptible_individuals[will_infected]
            
            for idx in will_infected_individuals.index:
                population.at[idx, 'state'] = 'Exposed'
                population.at[idx, 's_time_exposed'] = t
                population.at[idx, 's_time_infectious'] = t + rng.exponential(p.exposed_duration)
                population.at[idx, 's_time_recovery'] = population.at[idx, 's_time_infectious'] + rng.exponential(p.inf_duration)
                population.at[idx, 'exposed_from'] = random.choice(infected_ids)
            
            # I -> R state transition
            population.loc[(population["state"] == "Infectious") & (population["s_time_recovery"] < t), "state"] = "Recovered"
            
            # E -> I state transition
            population.loc[(population["state"] == "Exposed") & (population["s_time_infectious"] < t), "state"] = "Infectious"
            
            new_infs_from_seed = population[(population["s_time_exposed"] == t) & (population["exposed_from"] == seed_infection_index)]
            if p.record_all_new_cases:
                new_exposed_cases = population[population["s_time_exposed"] == t]
            
            if not new_infs_from_seed.empty:
                exposed_by_seed_df = pd.concat([exposed_by_seed_df, new_infs_from_seed], ignore_index=True)
            secondary_infections_from_seed_infection += len(new_infs_from_seed)
            
            if p.record_all_new_cases and not new_exposed_cases.empty:
                all_exposed_cases = pd.concat([all_exposed_cases, new_exposed_cases], ignore_index=True)
            
            if p.record_transmission:
                cur_records = possible_states.copy()
                cur_records['count'] = population['state'].value_counts().reindex(cur_records['state']).fillna(0).values
                cur_records['t'] = t
                cur_records['run_no'] = run
                all_records = pd.concat([all_records, cur_records], ignore_index=True)
        
        secondary_infections_from_seed_infection_list.append(secondary_infections_from_seed_infection)
    
    results = {
        "SAR": secondary_infections_from_seed_infection_list,
        "exposed_by_seed": exposed_by_seed_df
    }
    
    if p.record_transmission:
        results["all_transmission"] = all_records

    if p.record_all_new_cases:
        results["all_exposed_cases"] = all_exposed_cases
        
    return results


# Parameters
params = Params(
    no_runs=10,
    pop_size=5000,
    inf_duration=10, 
    exposed_duration=4,
    random_seed=5,
    transmission_rate=0.02,
    time_horizon=50,
    time_step=0.5,
    record_transmission=True,
    record_all_new_cases=True
)

##### Plot transmission
def plot_SEIR_pd(transmission_df, fig_size):
    fig, ax = plt.subplots(figsize=fig_size)
    
    for r in transmission_df["run_no"].unique():
        cur_df = transmission_df[transmission_df["run_no"] == r].sort_values("t")
        
        for state, color, label in zip(["Susceptible", "Infectious", "Exposed", "Recovered"],
                                       ["tab:green", "tab:red", "tab:orange", "tab:blue"],
                                       ["Susceptible", "Infectious", "Exposed", "Recovered"]):
            state_df = cur_df[cur_df["state"] == state]
            if not state_df.empty:
                times = state_df["t"].values
                counts = state_df["count"].values
                plt.plot(times, counts, color=color, label=label )#if r == transmission_df["run_no"].unique()[0] else "")
    
    plt.xlabel("Time (days)")
    plt.ylabel("Number")
    #plt.title("Disease Spread Dynamics")
    plt.grid()

    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()
    fig.savefig('plot_SEIR_pd.tiff', bbox_inches="tight", dpi=300)

if __name__ == "__main__":     
    # get the start time
    st = time.time()

    results = run_SEIR_model_pd(params)

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    print(f"Secondary infections from seed in each run: {results['SAR']}")


    plot_SEIR_pd(results["all_transmission"])
    
    #results["exposed_by_seed"].to_csv('exposed_by_seed.csv', index=False)
    #results["all_transmission"].to_csv('transmission.csv', index=False)
    #results["all_exposed_cases"].to_csv('all_exposed_cases.csv', index=False)


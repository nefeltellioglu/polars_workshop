#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:19:01 2024

@author: nefeltellioglu
"""

import polars as pl
import numpy as np
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)

import time



@dataclass
class Params:
    no_runs: int  
    pop_size: int 
    inf_duration: float
    exposed_duration:float 
    random_seed: int 
    transmission_rate: float
    time_horizon: float
    time_step: float
    record_transmission: bool
    record_all_new_cases: bool


def run_SEIR_model_pl(p: Params):
    secondary_infections_from_seed_infection_list = []
    exposed_by_seed_df = pl.DataFrame()
    all_exposed_cases = pl.DataFrame()
    rng = np.random.RandomState(p.random_seed)
    ts = np.arange(p.time_step, p.time_horizon, p.time_step)    
    transmission_rate = 1 - ((1 - p.transmission_rate)**(p.time_step))
    
    if p.record_transmission:
        possible_states = pl.DataFrame(
           { "state": ["Susceptible", "Exposed", "Infectious", "Recovered"],
            "count":  [0,0,0,0]}
            )
        all_records = pl.DataFrame()
        
    for run in range(p.no_runs):
        secondary_infections_from_seed_infection = 0
        
        # Initialize population
        population = pl.DataFrame(
        {
        "id": range(p.pop_size),
        "state": ["Susceptible"] * p.pop_size,
        "s_time_exposed": [0] * p.pop_size,
        "s_time_infectious": [0] * p.pop_size,
        "s_time_recovery": [0] * p.pop_size,
        "exposed_from": [-1] * p.pop_size,
        "pop_size": p.pop_size,
        #"hh_id": run,
        "run_no": run,
        }
        )
        
        # Initialize population
        cur_exposed_by_seed_df = pl.DataFrame()
        cur_all_exposed_cases = pl.DataFrame()
        
        # Infect one individual in the population (seed infection)
        
        seed_infection_index = 0#rng.randint(0, p.pop_size - 1)
        population = population.with_columns(
        (pl.when(pl.col("id") == seed_infection_index)
        .then(pl.lit("Infectious"))
        .otherwise(pl.col("state"))).alias("state"),
        
        (pl.when(pl.col("id") == seed_infection_index)
        .then(rng.exponential(p.inf_duration))
        .otherwise(pl.col("s_time_recovery"))).alias("s_time_recovery"),
        )
        
        
        if p.record_transmission:
            cur_records = possible_states.update(
                population.group_by(pl.col("state")).agg(
                pl.count()), on = ["state"], how = "left").with_columns(
                    pl.lit(ts[0] - p.time_step).alias("t"))
            all_records =  all_records.vstack(cur_records.with_columns(
                pl.Series("run_no", [run] * cur_records.height),
                ))
        
        
        # Simulate transmission in the population
        for t in ts:
            infected_ids = population.filter(
                pl.col("state") == "Infectious")["id"]
            susceptible_individuals = population.filter(
                pl.col("state") == "Susceptible")
            will_infected_individuals = susceptible_individuals.with_columns(
                 pl.Series(rng.rand(susceptible_individuals.height) 
                     < (transmission_rate * len(infected_ids)))
                 .alias("will_infected")
                    ).filter(pl.col("will_infected")).drop("will_infected")
            s_time_infectious = pl.Series("s_time_infectious", 
                                t + rng.exponential(p.exposed_duration,
                                    will_infected_individuals.height) ) 
                                
            will_infected_individuals = will_infected_individuals.with_columns(
                pl.lit(t).alias("s_time_exposed"),
                pl.lit(s_time_infectious).alias("s_time_infectious"),
                pl.Series("s_time_recovery", 
                        s_time_infectious + rng.exponential(p.inf_duration,
                                        will_infected_individuals.height)),
                pl.Series("exposed_from", 
                         rng.choice(infected_ids, 
                                    size = will_infected_individuals.height)),
                pl.lit("Exposed").alias("state"),
                )
            
            population = population.update(will_infected_individuals, on = "id", how= "left")
            
            #I -> R state transition
            population = population.with_columns(
                pl.when((pl.col("state") == "Infectious" ) & 
                        ( pl.col("s_time_recovery") < t))
                .then(pl.lit("Recovered")).otherwise(pl.col("state"))
                .alias("state"),
                )
            
            #E -> I state transition
            population = population.with_columns(
               pl.when((pl.col("state") == "Exposed" ) & 
                       (pl.col("s_time_infectious") < t))
                .then(pl.lit("Infectious")).otherwise(pl.col("state"))
                .alias("state"),
                )
            
            
            new_infs_from_seed = population.filter(
                            (pl.col("s_time_exposed") == t) &
                             (pl.col("exposed_from") == seed_infection_index)
                             )
            if p.record_all_new_cases:
                new_exposed_cases = population.filter(
                                (pl.col("s_time_exposed") == t))
                
            cur_exposed_by_seed_df = cur_exposed_by_seed_df.vstack(new_infs_from_seed)
            secondary_infections_from_seed_infection += new_infs_from_seed.height
            
            #record all new exposed cases
            cur_all_exposed_cases = cur_all_exposed_cases.vstack(new_exposed_cases)
            #record transmissions
            if p.record_transmission:
                cur_records =  cur_records.vstack(possible_states.update(
                    population.group_by(pl.col("state")).agg(
                    pl.count()), on = ["state"], how = "left").with_columns(
                        pl.lit(t).alias("t")))
            
        secondary_infections_from_seed_infection_list.append(secondary_infections_from_seed_infection)
        if cur_exposed_by_seed_df.height:
            #cur_exposed_by_seed_df = cur_exposed_by_seed_df.with_columns(
            #    pl.Series("run_no", [run] * cur_exposed_by_seed_df.height))
            exposed_by_seed_df = exposed_by_seed_df.vstack(cur_exposed_by_seed_df)
        if p.record_all_new_cases and cur_all_exposed_cases.height:
            #cur_all_exposed_cases = cur_all_exposed_cases.with_columns(
            #    pl.Series("run_no", [run] * cur_all_exposed_cases.height))
            all_exposed_cases = all_exposed_cases.vstack(cur_all_exposed_cases)
        
        
        
        if p.record_transmission:
            all_records =  all_records.vstack(cur_records.with_columns(
                pl.Series("run_no", [run] * cur_records.height),
                ))
    
    results = {"SAR": secondary_infections_from_seed_infection_list, 
                  "exposed_by_seed": exposed_by_seed_df}
    if p.record_transmission:
        results["all_transmission"] = all_records

    if p.record_all_new_cases:
        results["all_exposed_cases"] = all_exposed_cases
        
    return results


#TODO: parameters need to be calibrated for covid
#If SAR for covid is provided
#I can calibrate the transmission rate so that the SAR would fall between 
#COVID SAR  intervals
params = Params(no_runs = 10,
                pop_size = 5000,
                inf_duration = 10, 
                exposed_duration = 4,
                random_seed = 5,
                transmission_rate = 0.02,
                time_horizon = 50,
                time_step = 0.5,
                record_transmission= True,
                record_all_new_cases = True)



#####plot transmission
def plot_SEIR_pl(transmission_df, fig_size) -> None:
    
    
    fig, ax = plt.subplots(figsize= fig_size)
    unique_runs = transmission_df["run_no"].unique().to_list()

    for r in unique_runs:
        cur_df = transmission_df.filter(pl.col("run_no") == r).sort("t")

        for state, color, label in zip(["Susceptible", "Infectious", "Exposed", "Recovered"],
                                       ["tab:green", "tab:red", "tab:orange", "tab:blue"],
                                       ["Susceptible", "Infectious", "Exposed", "Recovered"]):
            state_df = cur_df.filter(pl.col("state") == state)
            plt.plot(state_df["t"].to_numpy(), state_df["count"].to_numpy(), color=color, label=label)# if r == unique_runs[0] else "")

    plt.xlabel("Time (days)")
    plt.ylabel("Number")
    #plt.title("Disease Spread Dynamics")
    plt.grid()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()
    fig.savefig('plot_SEIR_pl.tiff', bbox_inches="tight", dpi=300)
    
if __name__ == "__main__":     
    # get the start time
    st = time.time()
    
    results = run_SEIR_model_pl(params)
    #secondary_infections_from_seed_infection_list, exposed_by_seed_df, all_records = run_SEIR_model(params)
    
    # get the end time
    et = time.time()
    
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')   
    
    print(f"Secondary infections from seed in each run: {results['SAR']}")
    results["exposed_by_seed"] 
    results["all_transmission"] 
    results["all_exposed_cases"] 
    
    plot_SEIR_pl(results["all_transmission"])
    
    #results["exposed_by_seed"].write_csv(os.path.join('exposed_by_seed.csv'))
    #results["all_transmission"].write_csv(os.path.join('transmission.csv'))
    #results["all_exposed_cases"].write_csv(os.path.join('all_exposed_cases.csv'))
       

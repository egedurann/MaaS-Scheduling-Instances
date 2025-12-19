
from docplex.mp.progress import ProgressListener
from docplex.mp.model import *
from data import data_read
import numpy as np
import pandas as pd
import time
import csv
from openpyxl import Workbook
wb = Workbook()
wb.remove(wb.active)
np.random.seed(0)


instances,s,tc,machines=data_read()

class SolutionTracker(ProgressListener):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        self.improvements = None
        self.last_objective = None

    def notify_progress(self, data):
        if data.has_incumbent:
            current_obj = data.current_objective
            if self.last_objective is None or abs(current_obj - self.last_objective) > 1e-4:
                t = time.time() - self.start_time
                self.improvements=t
                self.last_objective = current_obj

def math_model(job_list,setup,machines,tc,dc):
    job_number = len(job_list)

    mdl = Model("MAS")
    m=[k for k in range(1,machines+1)]
    i_k=[(i,k) for i in range(1,job_number+1) for k in m]
    i_j_k=[(i,j,k) for i in range(job_number+1) for j in range(job_number+1) for k in m]
    i_k_l = [(i, k,l) for i in range(1, job_number + 1) for k in m for l in range(1,dc+1)]
    i_l=[(i,l) for i in range(1,job_number+1) for l in range(1,dc+1)]

    v = mdl.integer_var_dict(i_k, lb=0, name="v")
    c = mdl.integer_var_dict(i_k, name="c")
    T = {i: mdl.integer_var(name=f"T_{i}") for i in range(1, job_number + 1)}
    x = mdl.binary_var_dict(i_k, lb=0, ub=1, name="x")
    y = mdl.binary_var_dict(i_j_k, lb=0, ub=1, name="y")
    z=mdl.binary_var_dict(i_k_l, lb=0, ub=1, name="z")
    z_tot=mdl.binary_var_dict(i_l,lb=0, ub=1,name="z_tot")
    c_b = mdl.integer_var(name="c_b")
    F = mdl.continuous_var(name="F")
    M = 1000

    mdl.minimize(F)

    mdl.add_constraint(F >= 0.8 * (mdl.sum(T[i] for i in range(1,job_number+1) if job_list[i-1].agent=="A")) + 0.2 * c_b)

    #ct1
    for k in m:
        mdl.add_constraint(mdl.sum(y[0,i,k] for i in range(1,job_number+1))==1)

    for i in range(1,job_number+1):
        if job_list[i-1].agent!="C":
            mdl.add_constraint(mdl.sum(v[i,k] for k in m)==job_list[i-1].amount)

    for i in range(1,job_number+1):
        if job_list[i - 1].agent != "C":
            for k in m:
                mdl.add_constraint(x[i,k]<=v[i,k])
                mdl.add_constraint(v[i, k] <= M * x[i, k])

    for i in range(1,job_number+1):
        if job_list[i - 1].agent != "C":
            for k in m:
                mdl.add_constraint(v[i,k]<=x[i,k]*job_list[i-1].amount)

    for i in range(1,job_number+1):
        for k in m:
            mdl.add_constraint(c[i,k]+M*(1-y[0,i,k])>=job_list[i-1].process_time[k-1]*v[i,k]+setup[k-1][0][i])

    # 7
    for j in range(1, job_number + 1):
        for k in m:
            mdl.add_constraint(mdl.sum(y[i, j, k] for i in range(job_number + 1) if i != j) == x[j, k])

    # 8
    for i in range(1, job_number + 1):
        for j in range(1, job_number + 1):
            for k in m:
                mdl.add_constraint(y[i, j, k] + y[j, i, k] <= x[i, k])

    for i in range(1,job_number+1):
        for j in range(1,job_number+1):
            for k in m:
                mdl.add_constraint(c[j,k]+M*(1-y[i,j,k])>=c[i,k]+job_list[j-1].process_time[k-1]*v[j,k]+setup[k-1][i][j])

    #for i in range(1,job_number+1):
    #    for j in range(1,job_number+1):
    #        if job_list[i-1].agent=="C" and i!=j:
    #            mdl.add_constraint(c[j,k]+M * (1 - y[j, i, k])<=c[i,k]-setup[k-1][j][i])

    # 10
    for i in range(1, job_number + 1):
        for k in m:
            if job_list[i-1].agent == "A":
                for l in range(1,dc+1):
                    mdl.add_constraint(T[i] >= c[i, k]+z[i,k,l]*tc[l-1][k-1]-job_list[i-1].due_date)
    # 11
    for i in range(1, job_number + 1):
        for k in m:
            if job_list[i-1].agent == "B":
                for l in range(1,dc+1):
                    mdl.add_constraint(c_b >= c[i, k]+z[i,k,l]*tc[l-1][k-1])

    # 12
    for i in range(1, job_number + 1):
        for k in m:
            mdl.add_constraint(mdl.sum(y[i, j, k] for j in range(1, job_number + 1) if i != j) <= 1)
    # 13
    for i in range(1, job_number + 1):
        for k in m:
            mdl.add_constraint(mdl.sum(y[j, i, k] for j in range(1, job_number + 1) if i != j) <= 1)

    for i in range(1,job_number+1):
        if job_list[i - 1].agent != "C":
            for k in m:
                mdl.add_constraint(mdl.sum(z[i,k,l] for l in range(1,dc+1))==x[i,k])

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for k in m:
                for l in range(1,dc+1):
                    mdl.add_constraint(z[i,k,l]<=x[i,k])

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for l in range(1, dc + 1):
                for k in m:
                    mdl.add_constraint(z[i, k, l] <= z_tot[i, l])

    for i in range(1, job_number + 1):
        if job_list[i-1].agent!="C":
            mdl.add_constraint(mdl.sum(z_tot[i, l] for l in range(1,dc+1))== 1)

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent == "C":
            for k in m:
                if job_list[i - 1].assigned_machine == k:
                    mdl.add_constraint(x[i, k] == 1)
                    mdl.add_constraint(c[i, k] == job_list[i - 1].completion_time)
                    mdl.add_constraint(v[i,k]==1)
                else:
                    mdl.add_constraint(x[i, k] == 0)
    return mdl,c_b,T

results=[]
for ins in instances:
    if instances.index(ins)+1==10:
    #if instances.index(ins)+1>10 and instances.index(ins)+1<21:
    #if instances.index(ins)+1>=21:
        print(instances.index(ins)+1)
        number_job=0
        order_no=0
        for i in ins:
            if i.agent!="C":
                number_job+=i.amount
                order_no+=1
        trd=0
        start = time.time()
        dc=len(tc[instances.index(ins)])
        mdl,c_b,T=math_model(ins,s[instances.index(ins)],machines[instances.index(ins)],tc[instances.index(ins)],dc)
        mdl.set_time_limit(3600)
        tracker = SolutionTracker()
        mdl.add_progress_listener(tracker)
        model=mdl.solve()

        end = time.time()
        run_time = end - start
        if model is not None:
            gap = model.solve_details.mip_relative_gap
            obj = model.objective_value
            ms=model[c_b]
            for i in range(len(ins)):
                if ins[i].agent=="A":
                    trd+=model[T[i+1]]
            best_time = tracker.improvements

        else:
            obj = "Not found"
            gap = "-"
            Tardiness = "-"
            ms="-"
            best_time='-'

        #print(obj)
        #mdl.print_solution()

        results.append({
            'Instance': instances.index(ins) + 1,
            'order': order_no,
            "job no": number_job,
            "assigned job": len(ins) - order_no,
            "Machine_no": machines[instances.index(ins)],
            "DC":dc,
            "Obj": obj,
            "Tardiness": trd,
            "Makespan": ms,
            'Total Solve Time': run_time,
            "Gap": gap,
            'Solve Time': best_time
        })


file_path = 'results_MIP_1.txt'
# Open the file in write mode and use CSV formatting
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')  # Use tab as a delimiter for better readability

    # Write the header row (definitions)
    writer.writerow(["Instance", "Order", "Job No",  "Machine No", "DC",
                     "Obj", "Tardiness", "Makespan", "Total Solve Time",'Solve Time','Gap'])

    # Write data rows
    for result in results:
        writer.writerow([
            result['Instance'], result['order'], result["job no"],
            result["Machine_no"], result["DC"], result["Obj"], result["Tardiness"],
            result["Makespan"], result['Total Solve Time'],result['Solve Time'],result['Gap']
        ])
print()
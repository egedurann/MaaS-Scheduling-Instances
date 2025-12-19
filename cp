import time
from docplex.mp.model import Model
from docplex.cp.solver.solver import CpoSolver
from docplex.cp.model import *
from data import data_read
import numpy as np
import csv
import pandas as pd
from itertools import chain

np.random.seed(0)


instances,s,tc,machines=data_read()

def CP_opt(job_list,setup_time,machines,tc_list,dc):
    job_number=len(job_list)
    job = {} #dict for interval variables
    u = {}  # dict for integer variables
    tar={} #dict for tardiness variable
    model = CpoModel(name="splitting")
    makespan=interval_var(name="makespan", start=(0, 2000), end=(0, 2000),size=0)
    c_b = integer_var(name="c_b", min=0, max=600)

    z = [integer_var(min=1, max=dc, name=f"z_{i}") for i in range(1,job_number+1)]

    for i in range(1,job_number+1):
        if job_list[i-1].agent=="A":
            tar[i]=integer_var(name='T_{}'.format(i), min=0, max=5000)

        for j in range(1,machines+1):
            job[i, j] = interval_var(name='x_{}_{}'.format(i, j), start=(0, 2000), end=(0, 2000), optional=True,
                             size=(0, job_list[i-1].total_time[j-1]))

            u[i, j] = integer_var(name='u_{}_{}'.format(i, j), min=0, max=job_list[i - 1].amount)


    for j in range(1,machines+1): #create dummy orders for each machine
        a = interval_var(name='x_{}_{}'.format(0, j), start=0, end=(0, 2000), size=0)
        job[0, j] = a

    for i in range(1,job_number+1):
        if job_list[i-1].agent!="C":
            for j in range(1,machines+1):
                model.add(job_list[i-1].process_time[j-1] * u[i, j] * presence_of(job[i, j]) == size_of(job[i, j]))
                model.add(u[i, j] >= presence_of(job[i, j]))
                model.add(presence_of(job[i, j]) * job_list[i-1].amount >= u[i, j])
                #model.add(u[i,j]==v[i,j].solution_value)
                #model.add(job_list[i-1].process_time * u[i,j] * presence_of(job[i, j]) == size_of(job[i, j]))
                model.add(u[i, j] >= presence_of(job[i, j]))
                #model.add(presence_of(job[i, j]) * job_list[i-1].amount >= u[i, j])
                #model.add_constraint(sum(z[i,j,l] for l in range(1,dc+1))==presence_of(job[i,j]))


    for i in range(1,job_number+1):
        if job_list[i-1].agent!="C":
            model.add(sum(u[i, j] * presence_of(job[i, j]) for j in range(1,machines+1)) == job_list[i-1].amount)

    sequence = {}
    for j in range(1,machines+1):
        seq = sequence_var([job[i, j] for i in range(job_number + 1)])
        sequence[j] = seq
        model.add_constraint(no_overlap(seq, setup_time[j-1],1))
        model.add(first(seq, job[0, j]))

    for i in range(1,job_number+1):
        if job_list[i-1].agent=="A":
            for j in range(1,machines+1):
                model.add(tar[i] >= end_of(job[i, j])+presence_of(job[i,j])*element(tc_list,(z[i-1]-1)*machines+j-1) - job_list[i-1].due_date)

    for i in range(1,job_number+1):
        if job_list[i-1].agent == "B":
            for j in range(1,machines+1):
                #model.add(end_before_start(job[i,j],makespan))
                model.add(c_b>=end_of(job[i,j])+presence_of(job[i,j])*element(tc_list,(z[i-1]-1)*machines+j-1))


    for i in range(1,job_number+1):
        if job_list[i - 1].agent == "C":
            for j in range(1, machines + 1):
                if job_list[i - 1].assigned_machine==j:
                    model.add(presence_of(job[i,j])==1)
                    model.add(end_of(job[i,j])==job_list[i - 1].completion_time)
                    model.add(presence_of(job[i,j])*job_list[i-1].process_time[j-1]==size_of(job[i,j]))
                else:
                    model.add(presence_of(job[i, j]) == 0)

    p1 = search_phase(
        vars=u.values(),
        varchooser=select_smallest(domain_size()),
        valuechooser=select_smallest(value_impact()))

    p2 = search_phase(
        vars=job.values())
    #search_array = [p1]
    #mdl.set_search_phases(search_array)

    model.minimize(0.8*model.sum(tar[i] for i in range(1,job_number+1) if job_list[i-1].agent=="A")+0.2*c_b)
    return model,tar,c_b

results=[]

for ins in instances:

    number_job=0
    order_no=0
    for i in ins:
        if i.agent!="C":
            number_job+=i.amount
            order_no+=1
    trd = 0
    tc_list = list(chain.from_iterable(tc[instances.index(ins)]))
    dc = len(tc[instances.index(ins)])
    model, tar, c_b = CP_opt(ins, s[instances.index(ins)], machines[instances.index(ins)], tc_list, dc)

    best_time = None
    best_obj = float('inf')

    start = time.time()

    # Start search with time limit
    siter = model.start_search(TimeLimit=3600)

    for sol in siter:
        obj = sol.get_objective_value()
        if obj < best_obj - 1e-4:
            best_obj = obj
            best_time = round(sol.get_solve_time())
            # compute trd for best solution
            trd = 0
            for i in range(len(ins)):
                if ins[i].agent == "A":
                    trd += sol.get_value(tar[i + 1])
    last_result = siter.get_last_result()
    gap = siter.get_last_result().get_objective_gap()
    ms = last_result.get_value(c_b)
    end = time.time()
    run_time = round(end - start)

    results.append({
        'Instance': instances.index(ins) + 1,
        'order': order_no,
        "job no": number_job,
        "Assigned job": len(ins) - order_no,
        "Machine_no":machines[instances.index(ins)],
        "DC": dc,
        "Obj": obj,
        "Tardiness": trd,
        "Makespan": ms,
        'Total Solve Time': run_time,
        'Solve Time':best_time,
        "Gap":gap

    })



file_path = 'results_cp.txt'
# Open the file in write mode and use CSV formatting
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')  # Use tab as a delimiter for better readability

    # Write the header row (definitions)
    writer.writerow(["Instance", "Order", "Job No", "Assigned Job", "Machine No", "DC",
                     "Obj", "Tardiness", "Makespan",'Total Solve Time', "Solve Time",'Gap'])

    # Write data rows
    for result in results:
        writer.writerow([
            result['Instance'], result['order'], result["job no"], result["Assigned job"],
            result["Machine_no"], result["DC"], result["Obj"], result["Tardiness"],
            result["Makespan"],result['Total Solve Time'], result['Solve Time'],result['Gap']
        ])


#df = pd.DataFrame(results)
#df.to_excel("results_cp.xlsx", index=False)


import time
import csv
from hexaly.optimizer import HexalyOptimizer, HxCallbackType
import pandas as pd
import hexaly.optimizer
import numpy as np
from data import data_read

np.random.seed(0)


instances,s,tc,machines=data_read()

class CallbackExample:
    def __init__(self):
        self.last_best_value = float('inf')
        self.last_best_running_time = 0

    def my_callback(self, optimizer, cb_type):
        stats = optimizer.statistics
        sol = optimizer.solution
        if sol.status.name not in ["FEASIBLE", "OPTIMAL"]:
            print("⏳ No feasible solution yet.")
            return

        obj_val = optimizer.model.objectives[0].value
        current_time = stats.running_time

        if obj_val < self.last_best_value:
            self.last_best_value = obj_val
            self.last_best_running_time = current_time

def hexaly_opt(job_list,setup_time,machines,tc,dc):
    orders = len(job_list)
    optimizer = HexalyOptimizer()
    model = optimizer.model

    # Task intervals for each order on each machine
    tasks = [[model.interval(0, 2000) for i in range(machines)] for j in range(orders+1)]
    t=[model.int(0, 100) for j in range(1,orders+1) if job_list[j-1].agent=="A"]
    # Decision variable: Number of jobs assigned to each machine
    u = [[model.int(0, job_list[j-1].amount) for _ in range(machines)] for j in range(1,orders+1)]
    z=[model.int(1, dc) for i in range(1,orders+1) if job_list[i-1].agent!="C"]
    makespan=model.int(1,1000)
    tc_matrix=np.array(tc).T
    tc_matrix = tc_matrix.tolist()
    transit_time_array = model.array(tc_matrix)
    jobs_order = [model.list(orders + 1) for _ in range(machines)]
    # machines = model.array(jobs_order)
    # model.constraint(model.partition(machines))

    # Boolean: If an order is assigned to a machine
    is_assigned = [[model.contains(jobs_order[m], j) for j in range(orders + 1)] for m in range(machines)]
    tasks_array = model.array(tasks)
    task_setup_time = model.array(setup_time)
    # assigned_machine=model.array(assigned_machine)

    for m in range(machines):
        model.constraint(model.at(jobs_order[m], 0) == 0)
        model.constraint(is_assigned[m][0] == True)
        model.constraint(model.length(tasks[0][m]) == 0)
        model.constraint(model.start(tasks[0][m]) == 0)

    for i in range(1,orders+1):
        if job_list[i-1].agent=="C":
            for m in range(machines):
                if job_list[i-1].assigned_machine==m+1:
                    model.constraint(is_assigned[m][i] == True)
                    model.constraint(model.length(tasks[i][m]) == job_list[i-1].process_time[m])
                    model.constraint(model.end(tasks[i][m])==job_list[i - 1].completion_time)
                else:
                    model.constraint(is_assigned[m][i] == False)
                    model.constraint(model.length(tasks[i][m]) == 0)

    for m in range(machines):
        sequence = jobs_order[m]
        sequence_lambda = model.lambda_function(
            lambda i: model.start(tasks_array[sequence[i + 1]][m]) >= model.end(tasks_array[sequence[i]][m])
                      + model.at(task_setup_time,m, sequence[i], sequence[i + 1]))
        model.constraint(model.and_(model.range(0, model.count(sequence) - 1), sequence_lambda))

    # Ensure each order is assigned to machines according to its demand
    for j in range(1, orders + 1):
        if job_list[j - 1].agent != "C":
            model.constraint(
                model.sum(u[j-1][m] * is_assigned[m][j] for m in range(machines)) == job_list[j - 1].amount)
            for m in range(machines):
                model.constraint(model.length(tasks[j][m]) == job_list[j-1].process_time[m] * u[j-1][m] * is_assigned[m][j])
                model.constraint(is_assigned[m][j] * job_list[j - 1].amount >= u[j-1][m])
                model.constraint(u[j-1][m] >= is_assigned[m][j])

    for j in range(1, orders + 1):
        if job_list[j - 1].agent == "B":
            for m in range(machines):
                # Ensure you're correctly using model.at to index into the transit time array
                model.constraint(makespan >= model.end(tasks[j][m]) + is_assigned[m][j] * model.at(transit_time_array[m], z[j-1]-1) )


    for j in range(1,orders+1):
        if job_list[j-1].agent == "A":
            for m in range(machines):
                model.constraint(t[j-1]>=model.end(tasks[j][m])-job_list[j-1].due_date+is_assigned[m][j]*model.at(transit_time_array[m],z[j-1]-1))


    total_tardiness = model.sum(t[j - 1] for j in range(1, orders + 1) if job_list[j - 1].agent == "A")
    model.minimize(0.8*total_tardiness+0.2*makespan)
    model.close()
    cb = CallbackExample()
    optimizer.add_callback(HxCallbackType.TIME_TICKED, cb.my_callback)
    #optimizer.param.time_between_ticks = 1
    optimizer.param.time_limit = 3600
    optimizer.solve()
    best_time=cb.last_best_running_time

    tasks_values = [[tasks[j][m].value for m in range(machines)] for j in range(orders + 1)]
    tasks_start = [[tasks[j][m].value.start() for m in range(machines)] for j in range(orders + 1)]
    tasks_end=[[tasks[j][m].value.end() for m in range(machines)] for j in range(orders + 1)]
    t_values = [t[j-1].value for j in range(1,orders + 1) if job_list[j-1].agent=="A" ]
    is_assigned_values = [[is_assigned[m][j].value for j in range(orders + 1)] for m in
                          range(machines)]
    z_values = [z[j-1].value for j in range(1,orders+1) if job_list[j-1].agent!="C"]
    makespan=makespan.value
    obj_val = optimizer.model.objectives[0].value
    sol = optimizer.solution
    gap = sol.get_objective_gap(0)
    return model, t_values, tasks_values, is_assigned_values, z_values,tasks_start,tasks_end,makespan,best_time,obj_val,gap

results=[]
for ins in instances:
    #if instances.index(ins) + 1 <= 10:
    #if instances.index(ins)+1>10 and instances.index(ins)+1<21:
    #if instances.index(ins) + 1 >= 21:
    #if instances.index(ins) + 1 ==1:
        trd = 0
        start=time.time()
        dc=len(tc[instances.index(ins)])
        model, t_values, tasks_values, is_assigned_values, z_values,tasks_start,tasks_end,makespan,best_time,obj_val,gap=hexaly_opt(ins,s[instances.index(ins)],machines[instances.index(ins)],tc[instances.index(ins)],dc)
        end=time.time()
        run_time = end - start
        number_job = 0
        order_no = 0
        for j in range(1,len(ins) + 1):
            if ins[j-1].agent=="A":
                trd+=t_values[j-1]
            if ins[j-1] != "C":
                number_job += ins[j-1].amount
                order_no += 1

        results.append({
            'Instance': instances.index(ins) + 1,
            'order': order_no,
            "job no": number_job,
            "Assigned job": len(ins) - order_no,
            "Machine_no": machines[instances.index(ins)],
            "DC": dc,
            "Obj": obj_val,
            "Tardiness": trd,
            "Makespan": makespan,
            'Total Solve Time': run_time,
            'Solve Time': best_time,
            "Gap": gap

        })


file_path = 'C:/Users/eduran/OneDrive - University College Cork\Desktop/exact_methods/results_hexaly.txt'

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


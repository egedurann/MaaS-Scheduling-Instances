import numpy as np
import csv
from docplex.mp.model import Model
from docplex.cp.model import *
from data import data_read
import numpy as np
import pandas as pd
from openpyxl import Workbook

wb = Workbook()
wb.remove(wb.active)

np.random.seed(0)
machines = 6

instances, s, tc = data_read(machines)


def math_model(job_list, setup, machines, tc,dc):
    job_number = len(job_list)
    mdl = Model("MAS")
    m = [k for k in range(1, machines + 1)]
    i_k = [(i, k) for i in range(1, job_number + 1) for k in m]
    i_j_k = [(i, j, k) for i in range(job_number + 1) for j in range(job_number + 1) for k in m]
    i_k_l = [(i, k, l) for i in range(1, job_number + 1) for k in m for l in range(1, dc + 1)]
    i_n_k = []
    i_l = [(i, l) for i in range(1, job_number + 1) for l in range(1, dc + 1)]

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for n in range(job_list[i - 1].amount):
                for k in m:
                    i_n_k.append((i, n + 1, k))

    v = mdl.integer_var_dict(i_k, lb=0, name="v")
    c = mdl.integer_var_dict(i_k, name="c")
    T = {i: mdl.integer_var(name=f"T_{i}") for i in range(1, job_number + 1)}
    x = mdl.binary_var_dict(i_k, lb=0, ub=1, name="x")
    y = mdl.continuous_var_dict(i_j_k, lb=0, ub=1, name="y")
    c_b = mdl.integer_var(name="c_b")
    ben = mdl.binary_var_dict(i_n_k, lb=0, ub=1, name="ben")
    z = mdl.binary_var_dict(i_k_l, lb=0, ub=1, name="z")
    z_tot = mdl.binary_var_dict(i_l, lb=0, ub=1, name="z_tot")
    F = mdl.continuous_var(name="F")
    M = 1000

    mdl.minimize(F)

    mdl.add_constraint(
        F >= 0.8 * (mdl.sum(T[i] for i in range(1, job_number + 1) if job_list[i - 1].agent == "A")) + 0.2 * c_b)

    # ct1
    for k in m:
        mdl.add_constraint(mdl.sum(y[0, i, k] for i in range(1, job_number + 1)) == 1)

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            mdl.add_constraint(mdl.sum(v[i, k] for k in m) == job_list[i - 1].amount)

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for k in m:
                mdl.add_constraint(x[i, k] <= v[i, k])
                mdl.add_constraint(v[i, k] <= M * x[i, k])

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for k in m:
                mdl.add_constraint(v[i, k] <= x[i, k] * job_list[i - 1].amount)

    for i in range(1, job_number + 1):
        for k in m:
            mdl.add_constraint(
                c[i, k] + M * (1 - y[0, i, k]) >= job_list[i - 1].process_time[k - 1] * v[i, k] + setup[k - 1][0][i])

    # 7
    for j in range(1, job_number + 1):
        for k in m:
            mdl.add_constraint(mdl.sum(y[i, j, k] for i in range(job_number + 1) if i != j) == x[j, k])

    # 8
    for i in range(1, job_number + 1):
        for j in range(1, job_number + 1):
            for k in m:
                mdl.add_constraint(y[i, j, k] + y[j, i, k] <= x[i, k])

    # 9
    for i in range(1, job_number + 1):
        for j in range(1, job_number + 1):
            for k in m:
                mdl.add_constraint(
                    c[j, k] + M * (1 - y[i, j, k]) >= c[i, k] + job_list[j - 1].process_time[k - 1] * v[j, k] +
                    setup[k - 1][i][j])

    # 10
    for i in range(1, job_number + 1):
        for k in m:
            if job_list[i - 1].agent == "A":
                for l in range(1, dc + 1):
                    mdl.add_constraint(T[i] >= c[i, k] + z[i, k, l] * tc[l - 1][k - 1] - job_list[i - 1].due_date)
    # 11
    for i in range(1, job_number + 1):
        for k in m:
            if job_list[i - 1].agent == "B":
                for l in range(1, dc + 1):
                    mdl.add_constraint(c_b >= c[i, k] + z[i, k, l] * tc[l - 1][k - 1])

    # 12
    for i in range(1, job_number + 1):
        for k in m:
            mdl.add_constraint(mdl.sum(y[i, j, k] for j in range(1, job_number + 1) if i != j) <= 1)
    # 13
    for i in range(1, job_number + 1):
        for k in m:
            mdl.add_constraint(mdl.sum(y[j, i, k] for j in range(1, job_number + 1) if i != j) <= 1)

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for k in m:
                for n in range(1, job_list[i - 1].amount + 1):
                    mdl.add_constraint(v[i, k] <= n + M * (1 - ben[i, n, k]))
                    mdl.add_constraint(v[i, k] >= n - M * (1 - ben[i, n, k]))
                    mdl.add_constraint(ben[i, n, k] <= x[i, k])
                mdl.add_constraint(x[i, k] <= mdl.sum(ben[i, n, k] for n in range(1, job_list[i - 1].amount + 1)))

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            mdl.add_constraint(mdl.sum(x[i, k] for k in m) == mdl.sum(
                ben[i, n, k] for n in range(1, job_list[i - 1].amount + 1) for k in m))

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for k in m:
                mdl.add_constraint(mdl.sum(ben[i, n, k] for n in range(1, job_list[i - 1].amount + 1)) <= 1)

        # for i in range(1, jobs_no + 1):
        #    for k in m:
        #        mdl.add_constraint(c[i, k] >= mdl.sum(ben[i, n, k] * n for n in range(1, order_no[i-1] + 1)) *process_time[i - 1])

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for k in m:
                mdl.add_constraint(c[i, k] >= mdl.sum(ben[i, n, k] * n for n in range(1, job_list[i - 1].amount + 1)) *
                                   job_list[i - 1].process_time[k - 1])

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for k in m:
                mdl.add_constraint(mdl.sum(z[i, k, l] for l in range(1, dc + 1)) == x[i, k])

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for k in m:
                for l in range(1, dc + 1):
                    mdl.add_constraint(z[i, k, l] <= x[i, k])

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for l in range(1, dc + 1):
                for k in m:
                    mdl.add_constraint(z[i, k, l] <= z_tot[i, l])

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            mdl.add_constraint(mdl.sum(z_tot[i, l] for l in range(1, dc + 1)) == 1)

    for i in range(1, job_number + 1):
        if job_list[i - 1].agent == "C":
            for k in m:
                if job_list[i - 1].assigned_machine == k:
                    mdl.add_constraint(x[i, k] == 1)
                    mdl.add_constraint(c[i, k] == job_list[i - 1].completion_time)
                    mdl.add_constraint(v[i, k] == 1)
                else:
                    mdl.add_constraint(x[i, k] == 0)

    return mdl, x, F, v, ben, i_n_k, c_b, T, z_tot, z


def cp_model(job_list, setup, machines, v, x, z_tot, tc,dc):
    mdl = CpoModel(name="splitting")
    i_k = []
    # makespan=interval_var(name="makespan", start=(0, 50000), end=(0, 50000),size=0)
    w = integer_var(name="c_b", min=0, max=20000)
    job = {}  # dict for interval variables
    T = {}  # dict for tardiness variable
    u = {}  # dict for integer variables

    # mdl = CpoModel(name="splitting")
    # makespan = interval_var(name="makespan", start=(0, 2000), end=(0, 2000), size=0)
    jobs_no = len(job_list)
    for i in range(jobs_no + 1):
        for j in range(1, machines + 1):
            if i == 0:
                i_k.append((0, j))
            elif i > 0 and x[i, j].solution_value >= 0.99:
                i_k.append((i, j))

    for i in range(1, jobs_no + 1):
        t = integer_var(name='T_{}'.format(i), min=0, max=1000)
        T[i] = t
    # create dummy orders for each machine
    for i in i_k:
        if i[0] == 0:
            a = interval_var(name='job_{}'.format(i), start=0, end=(0, 2000), size=0)
            job[i] = a
        else:
            epsilon = v[i].solution_value - int(v[i].solution_value)
            a = interval_var(name='job_{}'.format(i), start=(0, 2000), end=(0, 2000),
                             size=job_list[i[0] - 1].process_time[i[1] - 1] * int(v[i].solution_value + epsilon))
            job[i] = a

    sequence = {}
    for j in range(1, machines + 1):
        seq = sequence_var([job[i, j] for i in range(jobs_no + 1) if (i, j) in i_k])
        remove = [i for i in range(jobs_no + 1) if (i, j) not in i_k]
        matrix = [row for i, row in enumerate(setup[j - 1]) if i not in remove]
        matrix = [[value for j, value in enumerate(row) if j not in remove] for row in matrix]
        sequence[j] = seq
        mdl.add_constraint(no_overlap(seq, matrix, 1))
        mdl.add(first(seq, job[0, j]))

    for i in range(1, jobs_no + 1):
        if job_list[i - 1].agent == "C":
            for j in range(1, machines + 1):
                if job_list[i - 1].assigned_machine == j:
                    mdl.add(end_of(job[i, j]) == job_list[i - 1].completion_time)

    for i in range(1, jobs_no + 1):
        if job_list[i - 1].agent == "A":
            for j in range(1, machines + 1):
                if (i, j) in i_k:
                    for l in range(1, dc + 1):
                        mdl.add(T[i] >= end_of(job[i, j]) + z_tot[i, l].solution_value * tc[l - 1][j - 1] * presence_of(
                            job[i, j]) - job_list[i - 1].due_date)

    for i in range(1, jobs_no + 1):
        for j in range(1, machines + 1):
            if (i, j) in i_k:
                if job_list[i - 1].agent == "B":
                    # mdl.add(end_before_start(job[i,j],makespan))
                    for l in range(1, dc + 1):
                        mdl.add(w >= end_of(job[i, j]) + z_tot[i, l].solution_value * tc[l - 1][j - 1] * presence_of(
                            job[i, j]))

    mdl.minimize(0.8 * (mdl.sum(T[i] for i in range(1, jobs_no + 1) if job_list[i - 1].agent == "A")) + 0.2 * w)
    p2 = search_phase(
        vars=job.values())
    search_array = [p2]
    mdl.set_search_phases(search_array)
    return mdl, T, w


results_benders = []
for ins in instances:
    #if instances.index(ins) + 1 <= 10:
    #if instances.index(ins)+1>10 and instances.index(ins)+1<21:
    #if instances.index(ins) + 1 >= 21:
        Time_Left = 3600
        init_obj = 10000
        init_tar = 0
        init_ms = 0
        instance_result = []
        sheet_name = f"Instance_{instances.index(ins) + 1}"
        ws = wb.create_sheet(title=sheet_name)
        dc=len(tc[instances.index(ins)])
        mdl, x, F, v, ben, i_n_k, c_b, T, z_tot, z = math_model(ins, s[instances.index(ins)], machines,
                                                                tc[instances.index(ins)],dc)
        stop = False
        start = time.time()
        count = 0
        while stop == False:
            count += 1
            master_start_time = time.time()
            if Time_Left > 0.4:
                mdl.set_time_limit(Time_Left)
                msol = mdl.solve()
                master_end_time = time.time()
                master_run_time = master_end_time - master_start_time
                ben_arr = []
                for i in i_n_k:
                    if ben[i].solution_value >= 0.95:
                        ben_arr.append(i)

                model, Tar, w = cp_model(ins, s[instances.index(ins)], machines, v, x, z_tot, tc[instances.index(ins)],dc)
                slave_start_time = time.time()
                # print(int(master_run_time))
                Time_Left = Time_Left - int(master_run_time)
            if Time_Left > 0.4:
                sl = model.solve(TimeLimit=Time_Left)
                slave_end_time = time.time()
                slave_run_time = slave_end_time - slave_start_time
                Time_Left = Time_Left - int(slave_run_time)
                F_vk = sl.get_objective_value()
                mdl.add_constraint(F >= F_vk * (1 - mdl.sum(1 - ben[i] for i in ben_arr)))

                warmstart = mdl.new_solution()
                for i in range(1, len(ins) + 1):
                    for k in range(1, machines + 1):
                        warmstart.add_var_value(x[i, k], x[i, k].solution_value)
                        warmstart.add_var_value(v[i, k], v[i, k].solution_value)
                        if ins[i - 1].agent != "C":
                            for n in range(1, ins[i - 1].amount):
                                warmstart.add_var_value(ben[i, n, k], ben[i, n, k].solution_value)
                    if ins[i - 1].agent != "C":
                        for l in range(1, dc + 1):
                            warmstart.add_var_value(z_tot[i, l], z_tot[i, l].solution_value)
                            for k in range(1, machines + 1):
                                warmstart.add_var_value(z[i, k, l], z[i, k, l].solution_value)
                mdl.add_mip_start(warmstart)

                mdl.add_constraint(c_b >= sl[w] - mdl.sum(
                    (1 - ben[i, n, k]) * (v[i, k].solution_value * ins[i - 1].process_time[k - 1]) + max(
                        value for value in s[instances.index(ins)][k - 1][i] if value != 100)  for i in range(1, len(ins) + 1) if ins[i - 1].agent == "B" for
                    k in range(1, machines + 1) for n in range(1, ins[i - 1].amount + 1)))
                for i in range(1, len(ins) + 1):
                    if ins[i - 1].agent == "A":
                        mdl.add_constraint(T[i] >= sl[Tar[i]] - sl[Tar[i]] * mdl.sum(
                            (1 - ben[i, n, k]) for k in range(1, machines + 1) for n in
                            range(1, ins[i - 1].amount + 1)))
                    # elif ins[i - 1].agent == "B":
                    # mdl.add_constraint(c_b>=sl.get_var_solution(w)-mdl.sum((1-ben[i,n,k])*(v[i,k].solution_value*ins[i-1].process_time[k-1])+max(value for value in s[k - 1][instances.index(ins)][i] if value != 100) for k in range(1,machines+1) for n in range(1, ins[i-1].amount + 1)))
                    # mdl.add_constraint(c_b >= sl[w] - mdl.sum(
                    #    (1 - ben[i, n, k]) * (v[i, k].solution_value * ins[i - 1].process_time[k - 1]) + max(
                    #        value for value in s[instances.index(ins)][k - 1][i] if value != 100) + max(
                    #        tc[instances.index(ins)][k - 1]) for k in
                    #    range(1, machines + 1) for n in range(1, ins[i - 1].amount + 1)))
            else:
                F_vk = init_obj

            if F_vk < init_obj:
                init_obj = F_vk
                init_tar = 0
                for i in range(1, len(ins) + 1):
                    if ins[i - 1].agent == "A":
                        init_tar += sl[Tar[i]]
                init_ms = sl[w]
                init_count = count
                init_time = 3600 - Time_Left

            instance_result.append({
                'iteration': count,
                "slave": F_vk,
                "master": mdl.objective_value,
                'Master Solve Time': master_run_time,
                "Slave Solve Time": slave_run_time
            })

            # sl.print_solution()
            # model.export_as_cpo()

            print("master obj value:", mdl.objective_value)
            print("Slave obj value:", F_vk)
            print(Time_Left)
            # print(mdl.export_as_lp_string())
            if F_vk - 0.001 <= F.solution_value:
                F_vk=init_obj
                end = time.time()
                run_time = end - start
                stop = True
                trd = 0
                ms = sl[w]
                for i in range(len(ins)):
                    if ins[i].agent == "A":
                        trd += sl[Tar[i + 1]]
                results_benders.append({
                        'Instance': instances.index(ins) + 1,
                        "Machines": machines,
                        "DC": dc,
                        "Obj": F_vk,
                        "Tardiness:": trd,
                        "Makespan": ms,
                        'Solve Time': run_time,
                        "iteration": count,
                        "Run Time": run_time,
                        "Total Iteration": count
                })
                ws.append(["Iteration", "Slave Objective", "Master Objective", "Master Solve Time", "Slave Solve Time"])
                for res in instance_result:
                    ws.append([
                        res['iteration'],
                        res['slave'],
                        res['master'],
                        res['Master Solve Time'],
                        res['Slave Solve Time']
                    ])
                wb.save("detail_benders_1.xlsx")
                print(instances.index(ins) + 1)
                print("obj:", F_vk)
                print("iteration:", count)
                print("run_time:", run_time)

            elif Time_Left <= 0.00001:
                stop = True
                run_time=3600
                results_benders.append({
                    'Instance': instances.index(ins) + 1,
                    "Machines": machines,
                    "DC": dc,
                    "Obj": init_obj,
                    "Tardiness": init_tar,
                    "Makespan": init_ms,
                    'Solve Time': init_time,
                    "iteration": init_count,
                    "Run Time": run_time,
                    "Total Iteration": count
                })
                ws.append(["Iteration", "Slave Objective", "Master Objective", "Master Solve Time", "Slave Solve Time"])
                for res in instance_result:
                    ws.append([
                        res['iteration'],
                        res['slave'],
                        res['master'],
                        res['Master Solve Time'],
                        res['Slave Solve Time']
                    ])
                wb.save("detail_benders_1.xlsx")
                print(instances.index(ins) + 1)
                print("obj:", F_vk)
                print("iteration:", count)
                print("run_time:", 3600)
                print("**************")

file_path = '/home/eduran/PycharmProjects/pythonProject3/results_benders_SP1.txt'
# Open the file in write mode and use CSV formatting
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')  # Use tab as a delimiter for better readability

    # Write the header row (definitions)
    # Write the header row (definitions)
    writer.writerow(["Instance", "Machines", "DC", "Obj",
                     "Tardiness", "Makespan", "Solve Time", "iteration", "Run Time", 'Total Iteration'])

    # Write data rows
    for result in results_benders:
        writer.writerow([
            result['Instance'], result['Machines'], result["DC"], result["Obj"],
            result["Makespan"], result["Tardiness"],result["Solve Time"], result["iteration"],
            result["Run Time"], result['Total Iteration']
        ])






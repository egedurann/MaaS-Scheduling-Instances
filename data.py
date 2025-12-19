from numpy import random
import numpy as np
#np.random.seed(0)
import pandas as pd

class jobs:
    def __init__(self, job_id,amount,process_time,total_time,due_date,agent,assigned_machine,completion_time):
        self.job_id = job_id
        self.amount = amount
        self.process_time = process_time
        self.total_time = total_time
        self.due_date = due_date
        self.agent = agent
        self.assigned_machine=assigned_machine
        self.completion_time=completion_time

    def __repr__(self):
        return F' job_id: {self.job_id} '

    def int(self):
        return self.job_id

def data_read():
    instances = []
    tc = []
    setup_instances = []
    machines=[]
    order= 46
    dc=[[1,2,3],[1,2,3]]
    job_no=10
    for j in range(2):
        machine_no=2
        for z in range(3):
            for i in range(10):
                scheduled_jobs = np.random.randint(low=1, high=int(order / machine_no) + 2, size=machine_no)
                scheduled_jobs = scheduled_jobs.tolist()
                id = 1
                total_job = 0
                makespan=0
                job_list = []
                for k in range(order):
                    total_job+=job_no
                    process_time=np.random.randint(low=2, high=5,size=machine_no)
                    total_time=job_no*process_time
                    makespan+=total_time
                    due_date = 0
                    if k<order/2:
                        agent="A"
                        job_id="{}_{}".format(k+1,agent)
                    else:
                        agent="B"
                        job_id = "{}_{}".format(id, agent)
                        id+=1
                    assigned_machine = 0
                    completion_time=0
                    job = jobs(job_id, job_no, process_time, total_time, due_date, agent,assigned_machine,completion_time)
                    job_list.append(job)
                for k in job_list:
                    if k.agent=="A":
                        k.due_date = np.random.randint(low=min(k.total_time)+20, high=(max(k.total_time)+40 )*(total_job / (machine_no)))
                p = 0
                s = 1
                for k in scheduled_jobs:
                    p += 1
                    CT = []
                    for l in range(k):
                        agent = "C"
                        job_id = "{}_{}".format(s, agent)
                        s += 1
                        job_no = 1
                        assigned_machine = p
                        process_time = np.random.randint(low=1, high=4, size=machine_no)
                        if len(CT) > 0:
                            # Ensure no overlap by setting the next job to start after the previous job's completion
                            start_time =  np.random.randint(low=CT[-1] + 5,high=(CT[-1] +5+ makespan[scheduled_jobs.index(k)] / machine_no))  # Random setup time after last completion
                            completion_time = start_time + process_time[scheduled_jobs.index(k)]
                        else:
                            # For the first job, no need to account for prior completion time
                            start_time = np.random.randint(5, high=(makespan[scheduled_jobs.index(k)] / machine_no)+5)
                            completion_time = start_time + process_time[scheduled_jobs.index(k)]
                        CT.append(completion_time)
                        total_time=process_time
                        due_date=0
                        job = jobs(job_id, job_no, process_time, total_time, due_date,
                                   agent, assigned_machine,completion_time)
                        job_list.append(job)

                instances.append(job_list)
                setup = []
                for i in range(machine_no):
                    setup_matrix = np.random.randint(low=1, high=4, size=(order+sum(scheduled_jobs)+1, order +sum(scheduled_jobs)+1 ))
                    setup_matrix[np.diag_indices_from(setup_matrix)] = 100
                    setup_matrix = setup_matrix.tolist()
                    for w in setup_matrix:
                        w[0] = 100
                    setup.append(setup_matrix)
                setup_instances.append(setup)

                tc_matrix=np.random.randint(low=30, high=60, size=(dc[j][z], machine_no))
                tc.append(tc_matrix)
                machines.append(machine_no)
            #machine_no+=2
        order += 4
        job_no+=4

    return instances,setup_instances,tc,machines

#np.random.seed(0)
#machines=6
#instances,s,tc=data_read(machines)

#print(len(instances))
#with pd.ExcelWriter("output.xlsx") as writer:
#    for ins_idx, ins in enumerate(instances):
#        # Prepare job data
#        data = []
#        for i in ins:
#            if i.agent!="C":
#                row = {
#                    "Job ID": i.job_id,
#                    "Due Date": i.due_date,
#                    "Assigned Machine": "None"
#                }
#            else:
#                row = {
#                   "Job ID": i.job_id,
#                    "Due Date": i.completion_time,
#                    "Assigned Machine": i.assigned_machine
#                }
            # Add each element of `total_time` as a separate column
#            for idx, time in enumerate(i.total_time):
#                row[f"pt {idx + 1}"] = time
#            data.append(row)

            # Convert job data to a DataFrame
#            df_jobs = pd.DataFrame(data)
#            start_row = 0
#            df_jobs.to_excel(writer, sheet_name=f"Instance {ins_idx + 1}", index=False, startrow=start_row)
#            df_matrix = pd.DataFrame(tc[instances.index(ins)])
#            startcol = len(df_jobs.columns) + 2  # Start the matrix two columns after the job data
#            df_matrix.to_excel(writer, sheet_name=f"Instance {ins_idx + 1}", index=True, startrow=0, startcol=startcol)

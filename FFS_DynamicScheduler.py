#==============================================================================
# Flexible Flow Shop (FFS) Dynamic Scheduling System
#==============================================================================
# All jobs share the same stage sequence: Stage1 -> Stage2 -> ... -> Stagek
# Each stage has multiple parallel machines with different processing times
#
# Features:
#   1. Static FFS-GA optimization scheduling
#   2. Dynamic event-driven scheduling (job arrival, machine breakdown)
#   3. Dispatching rules + rescheduling
#   4. Gantt chart visualization
#==============================================================================

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import copy

# ================================================================
# Part 1: FFS Data Structures and Problem Generation
# ================================================================

def generate_ffs_problem(n_jobs, n_stages, machines_per_stage, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    machine_start = 0
    stage_machines = []
    for n_m in machines_per_stage:
        stage_machines.append(list(range(machine_start, machine_start + n_m)))
        machine_start += n_m
    P = []
    for j in range(n_jobs):
        job_stages = []
        for s in range(n_stages):
            stage_times = np.random.randint(3, 15, size=machines_per_stage[s]).tolist()
            job_stages.append(stage_times)
        P.append(job_stages)
    return stage_machines, P, machine_start


def generate_dynamic_ffs(n_initial_jobs, n_stages, machines_per_stage, n_dynamic_jobs=5, seed=42):
    stage_machines, P_initial, total_machines = generate_ffs_problem(
        n_initial_jobs, n_stages, machines_per_stage, seed)
    np.random.seed(seed + 1)
    random.seed(seed + 1)
    dynamic_jobs = []
    for j in range(n_dynamic_jobs):
        arrival = np.random.randint(30, 80)
        job_times = []
        for s in range(n_stages):
            stage_times = np.random.randint(3, 15, size=machines_per_stage[s]).tolist()
            job_times.append(stage_times)
        dynamic_jobs.append({"arrival": arrival, "processing": job_times, "id": n_initial_jobs + j})
    breakdown_events = []
    n_breakdowns = np.random.randint(1, 3)
    for _ in range(n_breakdowns):
        machine = np.random.randint(0, total_machines)
        start = np.random.randint(20, 60)
        duration = np.random.randint(5, 15)
        breakdown_events.append({"machine": machine, "start": start, "duration": duration, "end": start + duration})
    return stage_machines, P_initial, total_machines, dynamic_jobs, breakdown_events


def print_problem_info(stage_machines, P, name=""):
    n_jobs = len(P)
    n_stages = len(stage_machines)
    total_machines = max(max(m) for m in stage_machines) + 1 if stage_machines else 0
    print("\n" + "="*50)
    print(name)
    print("="*50)
    print(f"Jobs: {n_jobs}, Stages: {n_stages}, Machines: {total_machines}")
    print(f"Machines per stage: {[len(m) for m in stage_machines]}")
    print(f"Stage machine ids: {stage_machines}")
    print(f"Total operations: {n_jobs * n_stages}")
    return n_jobs, n_stages, total_machines

# ================================================================
# Part 2: FFS Static GA Scheduling
# ================================================================

def create_ffs_ind(stage_machines, P, use_greedy=True):
    n_jobs = len(P)
    n_stages = len(stage_machines)
    MS = []
    if use_greedy:
        machine_load = [0] * (max(max(m) for m in stage_machines) + 1)
        for j in range(n_jobs):
            for s in range(n_stages):
                avail_machines = stage_machines[s]
                best_m = 0
                best_load = float('inf')
                for m_idx, m_id in enumerate(avail_machines):
                    load = machine_load[m_id] + P[j][s][m_idx]
                    if load < best_load:
                        best_load = load
                        best_m = m_idx
                MS.append(best_m)
                machine_load[avail_machines[best_m]] += P[j][s][best_m]
    else:
        for j in range(n_jobs):
            for s in range(n_stages):
                MS.append(random.randint(0, len(stage_machines[s]) - 1))
    OS = []
    remaining = [n_stages] * n_jobs
    while any(r > 0 for r in remaining):
        j = random.randint(0, n_jobs - 1)
        if remaining[j] > 0:
            OS.append(j)
            remaining[j] -= 1
    return MS + OS


def create_ffs_pop(stage_machines, P, pop_size):
    pop = []
    for i in range(pop_size):
        pop.append(create_ffs_ind(stage_machines, P, use_greedy=(i == 0)))
    return pop


def decode_ffs(chromosome, stage_machines, P):
    n_jobs = len(P)
    n_stages = len(stage_machines)
    n_ops = n_jobs * n_stages
    MS = chromosome[:n_ops]
    OS = chromosome[n_ops:]
    job_stage = [0] * n_jobs
    job_ready = [0] * n_jobs
    total_machines = max(max(m) for m in stage_machines) + 1
    machine_ready = [0] * total_machines
    schedule = []
    for job in OS:
        s = job_stage[job]
        if s >= n_stages:
            continue
        m_idx = MS[job * n_stages + s]
        machine_id = stage_machines[s][m_idx]
        p_time = P[job][s][m_idx]
        start = max(job_ready[job], machine_ready[machine_id])
        end = start + p_time
        schedule.append((job, s, machine_id, start, end))
        job_stage[job] += 1
        job_ready[job] = end
        machine_ready[machine_id] = end
    makespan = max(job_ready) if job_ready else 0
    return schedule, makespan


def cross_ffs(A, B, n_stages):
    n_ops = len(A) // 2
    n_jobs = n_ops // n_stages
    MS_A, MS_B = list(A[:n_ops]), list(B[:n_ops])
    rl, rr = sorted(random.sample(range(n_ops), 2))
    new_MS1 = MS_A[:rl] + MS_B[rl:rr+1] + MS_A[rr+1:]
    new_MS2 = MS_B[:rl] + MS_A[rl:rr+1] + MS_B[rr+1:]
    OS_A, OS_B = list(A[n_ops:]), list(B[n_ops:])
    job_set = list(set(OS_A))
    if len(job_set) <= 1:
        return A, B
    n_extract = random.randint(1, len(job_set) - 1)
    S1 = set(random.sample(job_set, n_extract))
    child1 = [None] * len(OS_A)
    temp = [x for x in OS_B if x not in S1]
    idx = 0
    for i in range(len(OS_A)):
        if OS_A[i] in S1:
            child1[i] = OS_A[i]
        else:
            child1[i] = temp[idx]
            idx += 1
    child2 = [None] * len(OS_B)
    temp = [x for x in OS_A if x not in S1]
    idx = 0
    for i in range(len(OS_B)):
        if OS_B[i] in S1:
            child2[i] = OS_B[i]
        else:
            child2[i] = temp[idx]
            idx += 1
    return new_MS1 + child1, new_MS2 + child2


def mutate_ffs(Ind, stage_machines, P, pm=0.1):
    n_jobs = len(P)
    n_stages = len(stage_machines)
    n_ops = n_jobs * n_stages
    MS = list(Ind[:n_ops])
    OS = list(Ind[n_ops:])
    idx = 0
    for j in range(n_jobs):
        for s in range(n_stages):
            if random.random() < pm:
                MS[idx] = random.randint(0, len(stage_machines[s]) - 1)
            idx += 1
    if random.random() < pm and len(OS) > 1:
        i1, i2 = random.sample(range(len(OS)), 2)
        rl, rr = min(i1, i2), max(i1, i2)
        OS[rl:rr] = OS[rl:rr][::-1]
    return MS + OS


def tournament_selection(fitness, k=3, pool=None):
    if pool is None:
        pool = len(fitness)
    n = len(fitness)
    result = []
    for _ in range(pool):
        indices = random.sample(range(n), k)
        best = min(indices, key=lambda i: fitness[i])
        result.append(best)
    return result


def ga_ffs(stage_machines, P, pop_size=80, generations=150, pc=0.8, pm=0.15, verbose=True):
    n_jobs = len(P)
    n_stages = len(stage_machines)
    total_machines = max(max(m) for m in stage_machines) + 1
    if verbose:
        print(f"\n{'='*50}")
        print("GA for FFS")
        print(f"{'='*50}")
        print(f"Jobs: {n_jobs}, Stages: {n_stages}, Machines: {total_machines}")
        print(f"Pop: {pop_size}, Gen: {generations}")
    pop = create_ffs_pop(stage_machines, P, pop_size)
    best_makespan = float('inf')
    best_chromosome = None
    best_schedule = None
    history = []
    for gen in range(generations):
        fitness = []
        schedules = []
        for chrom in pop:
            sched, mk = decode_ffs(chrom, stage_machines, P)
            fitness.append(mk)
            schedules.append(sched)
        gen_best = min(fitness)
        history.append(gen_best)
        if gen_best < best_makespan:
            best_makespan = gen_best
            idx = fitness.index(gen_best)
            best_chromosome = pop[idx].copy()
            best_schedule = schedules[idx]
        selected = tournament_selection(fitness, k=3, pool=pop_size)
        parents = [pop[i] for i in selected]
        new_pop = []
        i = 0
        while i < pop_size:
            if i + 1 < pop_size and random.random() < pc:
                c1, c2 = cross_ffs(parents[i], parents[i+1], n_stages)
                new_pop.append(c1)
                new_pop.append(c2)
                i += 2
            else:
                new_pop.append(parents[i].copy())
                i += 1
        for i in range(len(new_pop)):
            new_pop[i] = mutate_ffs(new_pop[i], stage_machines, P, pm)
        pop = new_pop
        if verbose and ((gen + 1) % 30 == 0 or gen == 0):
            print(f"  Gen {gen+1:4d}: best = {gen_best:3d}, global = {best_makespan:3d}")
    if verbose:
        print(f"\n  Final: Best Makespan = {best_makespan}")
    return best_schedule, best_makespan, best_chromosome, history

# ================================================================
# Part 3: Dispatching Rules
# ================================================================

def dispatch_schedule(stage_machines, P, rule='SPT'):
    n_jobs = len(P)
    n_stages = len(stage_machines)
    total_machines = max(max(m) for m in stage_machines) + 1
    machine_ready = [0] * total_machines
    job_ready = [0] * n_jobs
    job_stage = [0] * n_jobs
    remaining_work = []
    for j in range(n_jobs):
        total = 0
        for s in range(n_stages):
            total += min(P[j][s])
        remaining_work.append(total)
    schedule = []
    for s in range(n_stages):
        ops_in_stage = []
        for j in range(n_jobs):
            if job_stage[j] == s:
                for m_idx, m_id in enumerate(stage_machines[s]):
                    p = P[j][s][m_idx]
                    earliest_start = max(job_ready[j], machine_ready[m_id])
                    if rule == 'SPT':
                        priority = p
                    elif rule == 'MWKR':
                        priority = -remaining_work[j]
                    elif rule == 'LWKR':
                        priority = remaining_work[j]
                    elif rule == 'FIFO':
                        priority = j
                    else:
                        priority = p
                    ops_in_stage.append((priority, earliest_start, p, j, m_idx, m_id))
        ops_in_stage.sort(key=lambda x: (x[0], x[1]))
        for _, earliest_start, p, j, m_idx, m_id in ops_in_stage:
            if job_stage[j] != s:
                continue
            start = max(job_ready[j], machine_ready[m_id])
            end = start + p
            schedule.append((j, s, m_id, start, end))
            job_ready[j] = end
            machine_ready[m_id] = end
            job_stage[j] = s + 1
            remaining_work[j] -= min(P[j][s])
    makespan = max(job_ready) if job_ready else 0
    return schedule, makespan

# ================================================================
# Part 4: Dynamic Scheduling Engine
# ================================================================

class DynamicFFSEngine:
    def __init__(self, stage_machines, initial_P, schedule_rule='GA', reschedule_interval=20):
        self.stage_machines = stage_machines
        self.n_stages = len(stage_machines)
        self.total_machines = max(max(m) for m in stage_machines) + 1
        self.schedule_rule = schedule_rule
        self.reschedule_interval = reschedule_interval
        self.jobs = {}
        self.next_job_id = 0
        for j in range(len(initial_P)):
            self.jobs[j] = {'id': j, 'processing': initial_P[j], 'arrival': 0, 'stage': 0, 'ready_time': 0, 'active': True, 'dynamic': False}
            self.next_job_id = j + 1
        self.machine_status = {}
        for m in range(self.total_machines):
            self.machine_status[m] = {'available': True, 'available_at': 0, 'breakdown_until': 0}
        self.current_schedule = []
        self.future_schedule = []
        self.current_time = 0
        self.event_log = []
        self.makespan_history = []
        self.breakdown_events = []

    def add_job_arrival(self, arrival_time, processing):
        job_id = self.next_job_id
        self.next_job_id += 1
        self.jobs[job_id] = {'id': job_id, 'processing': processing, 'arrival': arrival_time, 'stage': 0, 'ready_time': arrival_time, 'active': True, 'dynamic': True}
        self.event_log.append({'time': arrival_time, 'type': 'job_arrival', 'job_id': job_id, 'desc': f'Job J{job_id} arrives at t={arrival_time}'})
        return job_id

    def add_breakdown(self, machine_id, start_time, duration):
        self.breakdown_events.append({'machine': machine_id, 'start': start_time, 'duration': duration, 'end': start_time + duration})
        self.event_log.append({'time': start_time, 'type': 'breakdown_start', 'machine': machine_id, 'desc': f'Machine M{machine_id} down at t={start_time} for {duration}'})

    def get_active_jobs(self):
        active = {}
        for jid, job in self.jobs.items():
            if job['active'] and job['stage'] < self.n_stages:
                active[jid] = job
        return active

    def build_processing_matrix(self, job_ids):
        P = []
        for jid in job_ids:
            P.append(self.jobs[jid]['processing'])
        return P

    def reschedule(self, current_time, affected_jobs=None):
        active_jobs = self.get_active_jobs()
        if not active_jobs:
            return []
        job_ids_to_reschedule = list(active_jobs.keys())
        if not job_ids_to_reschedule:
            return []
        P_resched = self.build_processing_matrix(job_ids_to_reschedule)
        for m in range(self.total_machines):
            bd_end = 0
            for bd in self.breakdown_events:
                if m == bd['machine'] and current_time < bd['end'] and current_time >= bd['start']:
                    bd_end = bd['end']
                    break
            if bd_end > 0:
                self.machine_status[m]['available'] = False
                self.machine_status[m]['available_at'] = bd_end
            elif current_time > self.machine_status[m]['available_at']:
                self.machine_status[m]['available'] = True
                self.machine_status[m]['available_at'] = current_time
        if self.schedule_rule == 'GA':
            sched, mk, _, _ = ga_ffs(self.stage_machines, P_resched, pop_size=40, generations=60, pm=0.15, verbose=False)
        else:
            sched, mk = dispatch_schedule(self.stage_machines, P_resched, rule=self.schedule_rule)
        mapped_schedule = []
        for job, stage, machine, start, end in sched:
            real_job_id = job_ids_to_reschedule[job]
            mapped_schedule.append((real_job_id, stage, machine, start + current_time, end + current_time))
        return mapped_schedule

    def run_simulation(self, max_time=200):
        self.current_time = 0
        print(f"\n{'='*60}")
        print(f"Dynamic Scheduling - Rule: {self.schedule_rule}")
        print(f"{'='*60}")
        print(f"Stages: {self.n_stages}, Machines: {self.total_machines}")
        initial_count = sum(1 for j in self.jobs.values() if not j['dynamic'])
        print(f"Initial jobs: {initial_count}")
        print(f"{'='*60}\n")

        print(f"[t={self.current_time}] Initial schedule...")
        self.future_schedule = self.reschedule(self.current_time)
        if self.future_schedule:
            self.current_makespan = max(end for _, _, _, _, end in self.future_schedule)
            print(f"[t={self.current_time}] Initial Makespan = {self.current_makespan}")
        else:
            self.current_makespan = 0
        self.makespan_history.append((self.current_time, self.current_makespan))

        next_reschedule = self.reschedule_interval
        all_events = []
        for jid, job in self.jobs.items():
            if job.get('dynamic') and job.get('arrival', 0) > 0:
                all_events.append({'time': job['arrival'], 'type': 'job_arrival', 'job_id': jid})
        for bd in self.breakdown_events:
            all_events.append({'time': bd['start'], 'type': 'breakdown', 'machine': bd['machine'], 'duration': bd['duration']})
        all_events.sort(key=lambda e: e['time'])

        trigger_reschedule = False
        event_idx = 0
        for current_time in range(0, max_time + 1):
            self.current_time = current_time
            while event_idx < len(all_events) and all_events[event_idx]['time'] <= current_time:
                event = all_events[event_idx]
                if event['type'] == 'job_arrival':
                    print(f"\n[Event t={current_time}] New job J{event['job_id']} arrives!")
                    trigger_reschedule = True
                elif event['type'] == 'breakdown':
                    print(f"\n[Event t={current_time}] Machine M{event['machine']} breaks down! Duration: {event['duration']}")
                    self.machine_status[event['machine']]['available'] = False
                    trigger_reschedule = True
                event_idx += 1
            if current_time >= next_reschedule:
                trigger_reschedule = True
                next_reschedule += self.reschedule_interval
            if trigger_reschedule:
                print(f"[t={current_time}] Rescheduling...")
                self.future_schedule = self.reschedule(current_time)
                if self.future_schedule:
                    self.current_makespan = max(end for _, _, _, _, end in self.future_schedule)
                    remaining = self.current_makespan - current_time
                    print(f"[t={current_time}] Remaining Makespan = {remaining}")
                trigger_reschedule = False
                self.makespan_history.append((current_time, self.current_makespan))

        completed = sum(1 for j in self.jobs.values() if j['stage'] >= self.n_stages)
        print(f"\n{'='*60}")
        print(f"Simulation done (t={max_time})")
        print(f"Completed: {completed}/{len(self.jobs)}")
        if self.future_schedule:
            final_mk = max(end for _, _, _, _, end in self.future_schedule)
            print(f"Final Makespan: {final_mk}")
        print(f"{'='*60}")
        return self.future_schedule

# ================================================================
# Part 5: Visualization
# ================================================================

def draw_ffs_gantt(schedule, stage_machines, n_jobs, title="FFS Gantt Chart"):
    if not schedule:
        print("No schedule data")
        return
    total_machines = max(max(m) for m in stage_machines) + 1
    makespan = max(end for _, _, _, _, end in schedule)
    colors = plt.cm.tab10(np.linspace(0, 1, n_jobs))
    fig, ax = plt.subplots(figsize=(16, 7))
    machine_ops = defaultdict(list)
    for job, stage, machine, start, end in schedule:
        machine_ops[machine].append((job, stage, start, end))
    for machine in range(total_machines):
        ops = sorted(machine_ops.get(machine, []), key=lambda x: x[2])
        for job, stage, start, end in ops:
            color = colors[job % n_jobs]
            ax.barh(machine, end - start, left=start, height=0.6, color=color, edgecolor='black', linewidth=0.8)
            ax.text((start + end) / 2, machine, f"J{job}S{stage}", ha='center', va='center', fontsize=8, fontweight='bold')
    for s_idx, machines in enumerate(stage_machines):
        if machines:
            mid = (min(machines) + max(machines)) / 2
            ax.text(-makespan * 0.02, mid, f"Stage{s_idx}", ha='right', va='center', fontsize=11, fontweight='bold')
    for s_idx in range(len(stage_machines) - 1):
        sep = max(stage_machines[s_idx]) + 0.5
        ax.axhline(y=sep + 0.5, xmin=0, xmax=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_yticks(range(total_machines))
    ax.set_yticklabels([f"M{m}" for m in range(total_machines)])
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Machine", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-makespan * 0.05, makespan * 1.05)
    ax.grid(axis='x', alpha=0.3)
    patches = [mpatches.Patch(color=colors[i], label=f"Job {i}") for i in range(min(n_jobs, 10))]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()


def draw_convergence(history, title="GA Convergence"):
    plt.figure(figsize=(10, 5))
    plt.plot(history, 'b-', linewidth=1.5)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Makespan", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def draw_makespan_timeline(makespan_history, title="Makespan Over Time"):
    times = [t for t, _ in makespan_history]
    values = [v for _, v in makespan_history]
    plt.figure(figsize=(12, 5))
    plt.step(times, values, where='post', linewidth=2, marker='o', markersize=6)
    plt.xlabel("Simulation Time", fontsize=12)
    plt.ylabel("Makespan", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ================================================================
# Part 6: Demo
# ================================================================

def run_static_demo():
    print("\n" + "="*60)
    print("  Static FFS Scheduling Demo")
    print("="*60)
    stage_machines, P, total_machines = generate_ffs_problem(n_jobs=5, n_stages=3, machines_per_stage=[2, 3, 2], seed=100)
    print_problem_info(stage_machines, P, "FFS Problem")
    print(f"\nProcessing times P[job][stage][machine]:")
    for j in range(len(P)):
        print(f"  J{j}: ", end="")
        for s in range(len(P[j])):
            print(f"  S{s}{P[j][s]}", end="")
        print()
    sched_ga, mk_ga, chrom, history = ga_ffs(stage_machines, P, pop_size=80, generations=120)
    print(f"\nGA Makespan = {mk_ga}")
    draw_convergence(history, "FFS-GA Convergence")
    draw_ffs_gantt(sched_ga, stage_machines, len(P), f"FFS-GA Schedule (Makespan={mk_ga})")
    print(f"\n{'='*40}")
    print("Dispatching Rules Comparison:")
    print(f"{'='*40}")
    for rule in ['SPT', 'MWKR', 'LWKR', 'FIFO']:
        sched, mk = dispatch_schedule(stage_machines, P, rule)
        print(f"  {rule:5s}: Makespan = {mk}")


def run_dynamic_demo():
    print("\n" + "="*60)
    print("  Dynamic FFS Scheduling Demo")
    print("="*60)
    stage_machines, P_initial, total_machines, dynamic_jobs, breakdowns =         generate_dynamic_ffs(n_initial_jobs=5, n_stages=3, machines_per_stage=[2, 3, 2], n_dynamic_jobs=3, seed=100)
    print_problem_info(stage_machines, P_initial, "Dynamic FFS")
    print(f"\nDynamic arriving jobs:")
    for dj in dynamic_jobs:
        print(f"  J{dj['id']}: arrival t={dj['arrival']}, times {[f'S{s}{t}' for s,t in enumerate(dj['processing'])]}")
    print(f"\nMachine breakdowns:")
    for bd in breakdowns:
        print(f"  M{bd['machine']}: t={bd['start']}~{bd['end']} (dur={bd['duration']})")

    print(f"\n{'='*40}")
    print("1) GA Dynamic Scheduling")
    print(f"{'='*40}")
    engine_ga = DynamicFFSEngine(stage_machines, P_initial, schedule_rule='GA', reschedule_interval=30)
    for dj in dynamic_jobs:
        engine_ga.add_job_arrival(dj['arrival'], dj['processing'])
    for bd in breakdowns:
        engine_ga.add_breakdown(bd['machine'], bd['start'], bd['duration'])
    final_sched_ga = engine_ga.run_simulation(max_time=150)
    if final_sched_ga:
        n_all = len(P_initial) + len(dynamic_jobs)
        final_mk = max(e for _,_,_,_,e in final_sched_ga)
        draw_ffs_gantt(final_sched_ga, stage_machines, n_all, f"GA Dynamic Schedule (Makespan={final_mk})")
    draw_makespan_timeline(engine_ga.makespan_history, "GA Makespan Over Time")

    print(f"\n{'='*40}")
    print("2) SPT Dynamic Scheduling (Comparison)")
    print(f"{'='*40}")
    engine_spt = DynamicFFSEngine(stage_machines, P_initial, schedule_rule='SPT', reschedule_interval=30)
    for dj in dynamic_jobs:
        engine_spt.add_job_arrival(dj['arrival'], dj['processing'])
    for bd in breakdowns:
        engine_spt.add_breakdown(bd['machine'], bd['start'], bd['duration'])
    final_sched_spt = engine_spt.run_simulation(max_time=150)
    if final_sched_spt:
        n_all = len(P_initial) + len(dynamic_jobs)
        final_mk = max(e for _,_,_,_,e in final_sched_spt)
        draw_ffs_gantt(final_sched_spt, stage_machines, n_all, f"SPT Dynamic Schedule (Makespan={final_mk})")
    draw_makespan_timeline(engine_spt.makespan_history, "SPT Makespan Over Time")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  Flexible Flow Shop (FFS) Dynamic Scheduling System")
    print("#"*60)
    run_static_demo()
    run_dynamic_demo()
    print("\n" + "="*60)
    print("  All demos completed")
    print("="*60)

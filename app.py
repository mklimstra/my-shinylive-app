import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny.express import input, output, render, ui

# Thelen muscle model with full calculations
def thelen_muscle(onoff, freq, excursion, L0, F0, Vx, af, tau_a, tau_d):
    try:
        onset = onoff[0] / 100
        offset = onoff[1] / 100
        excursion = excursion / 1000  # Convert mm to meters
        tau_a = tau_a / 1000  # Convert to seconds
        tau_d = tau_d / 1000  # Convert to seconds

        act_pct = 1
        onset_time = onset / freq
        offset_time = offset / freq
        dt = 0.001 / freq  # Time step
        V0 = Vx * L0
        penn0 = 0.087
        k_shape = 0.5
        w = L0 * np.sin(penn0)

        # Time and cycle percentage
        t = np.arange(0, 1.25 / freq, dt)
        cycle_pct = t * freq

        # Position
        position = excursion * np.sin(2 * np.pi * freq * t)
        position_mm = position * 1000

        # Excitation and Activation Dynamics
        excitation = np.where((t >= onset_time) & (t <= offset_time), 1 * act_pct, 0)
        da_dt = np.zeros(len(t))
        activation = np.zeros(len(t))

        for i in range(1, len(t)):
            if excitation[i - 1] >= activation[i - 1]:
                da_dt[i] = (excitation[i - 1] - activation[i - 1]) / tau_a
            else:
                da_dt[i] = (excitation[i - 1] - activation[i - 1]) / tau_d
            activation[i] = activation[i - 1] + da_dt[i] * dt

        # Pennation and muscle length
        penn = np.arcsin(w / (position + L0))
        muscle_length = position / np.cos(penn0) + L0
        muscle_length_norm = muscle_length / L0

        # Velocity
        v = (-2 * np.pi * freq * excursion * np.cos(2 * np.pi * freq * t)) / np.cos(penn)
        v_norm = v / (V0 * ((Vx / 2) + (Vx / 2) * activation) / Vx)

        # Force-Length relationship
        fl_norm = np.exp(-((muscle_length_norm - 1) ** 2) / k_shape)

        # Force-Velocity relationship
        fv_norm = np.where(v > (V0 * ((Vx / 2) + (Vx / 2))), 0,
                           np.where(v_norm > 0, (1 - v_norm) / (1 + v_norm / af),
                                    (1.8 - (0.8 * (1 + v / V0)) / (1 - 7.56 * 0.21 * v / V0))))

        # Contractile element force
        force_ce = activation * F0 * fl_norm * fv_norm

        # Total force
        force_total = force_ce

        # Work and Power calculations
        work = force_total * v * dt
        power = force_total * v
        work_actual = np.sum(work)
        work_positive = np.sum(work[work > 0])
        work_negative = np.sum(work[work < 0])
        power_actual = np.mean(power[(t >= onset_time) & (t <= onset_time + 1)])

        # Data to return
        sim_data = pd.DataFrame({
            't': t,
            'cycle_pct': cycle_pct,
            'position': position,
            'position_mm': position_mm,
            'velocity': v,
            'force_total': force_total,
            'work': work,
            'power': power,
            'excitation': excitation,
            'activation': activation
        })

        return {
            'sim_data': sim_data,
            'work_actual': work_actual,
            'work_positive': work_positive,
            'work_negative': work_negative,
            'power_actual': power_actual
        }
    except Exception as e:
        print(f"Error in thelen_muscle: {e}")
        return None

# Optimization function for the Thelen model
def thelen_muscle_opt(freq, excursion, L0, F0, Vx, af, tau_a, tau_d):
    onset_values = np.arange(20, 26, 1)
    offset_values = np.arange(26, 76, 1)

    best_onset = None
    best_offset = None
    max_power = -np.inf

    for onset in onset_values:
        for offset in offset_values:
            sim_results = thelen_muscle([onset, offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
            if sim_results is None:
                continue

            current_power = sim_results['power_actual']

            if current_power > max_power:
                best_onset = onset
                best_offset = offset
                max_power = current_power

    if best_onset is not None and best_offset is not None:
        optimized_results = thelen_muscle([best_onset, best_offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
        return optimized_results, best_onset, best_offset, max_power
    return None, None, None, None

# Define the run_simulation function with theoretical results
def run_simulation():
    # Simulated muscle parameters
    muscle_params = {
        "onoff": [input.onset(), input.offset()],
        "freq": input.cycle_freq(),
        "excursion": input.excursion(),
        "L0": input.length_optimal(),
        "F0": input.max_isometric_force(),
        "Vx": input.max_velocity(),
        "af": input.force_velocity_curvature(),
        "tau_a": input.activation_time(),
        "tau_d": input.deactivation_time(),
    }
    
    # Calculate simulated results
    if input.optimize():
        sim_results, _, _, _ = thelen_muscle_opt(
            input.cycle_freq(), input.excursion(), input.length_optimal(),
            input.max_isometric_force(), input.max_velocity(),
            input.force_velocity_curvature(), input.activation_time(),
            input.deactivation_time()
        )
    else:
        sim_results = thelen_muscle(**muscle_params)
    
    # Theoretical muscle parameters (instantaneous activation/deactivation and no onoff)
    theoretical_params = muscle_params.copy()
    theoretical_params['onoff'] = [25, 75]  # No onset or offset
    theoretical_params['tau_a'] = 0.5  # No activation time
    theoretical_params['tau_d'] = 1  # No deactivation time
    
    # Calculate theoretical results with zero onset/offset and instantaneous activation/deactivation
    theoretical_results = thelen_muscle(**theoretical_params)
    
    return sim_results, theoretical_results

# Define the UI layout
with ui.sidebar():
            ui.input_slider("onset", "Onset (% of cycle)", min=20, max=25, value=22),
            ui.input_slider("offset", "Offset (% of cycle)", min=26, max=100, value=66),
            ui.input_slider("excursion", "Excursion amplitude (mm)", min=1, max=50, value=20),
            ui.input_slider("cycle_freq", "Cycle frequency (Hz)", min=0.5, max=4.5, value=2.0),
            ui.input_numeric("length_optimal", "Length optimal (m)", value=0.084),
            ui.input_numeric("max_isometric_force", "Max isometric force (N)", value=1871),
            ui.input_numeric("max_velocity", "Max velocity (fiber lengths/s)", value=10),
            ui.input_numeric("force_velocity_curvature", "Force-velocity curvature", value=0.30),
            ui.input_numeric("activation_time", "Activation time (ms)", value=10),
            ui.input_numeric("deactivation_time", "Deactivation time (ms)", value=40),
            ui.input_checkbox("optimize", "Optimize Onset/Offset", value=False)


with ui.card():
    with ui.navset_bar(title="Virtual Muscle Lab"):
        with ui.nav_panel(title="Graphs"):
            @render.plot
            def combined_graphs():
                results = run_simulation()
                sim_results = results[0]  # First item in the tuple is sim_results
                theoretical_results = results[1]  # Second item is theoretical_results
                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                # Create a 2x2 grid of subplots
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns

                # Top-left: Force vs. % of Cycle
                force_total_sim = sim_results['sim_data']['force_total']
                cycle_pct_sim = sim_results['sim_data']['cycle_pct']
    
                force_total_theoretical = theoretical_results['sim_data']['force_total']
                cycle_pct_theoretical = theoretical_results['sim_data']['cycle_pct']
    
                axs[0, 0].plot(cycle_pct_sim, force_total_sim, label='Simulated Force')
                axs[0, 0].plot(cycle_pct_theoretical, force_total_theoretical, label='Theoretical Force', linestyle='--')
                axs[0, 0].set_title("Force vs. % of Cycle")
                axs[0, 0].set_xlabel("% of Cycle")
                axs[0, 0].set_ylabel("Force (N)")
                

                # Top-right: Velocity vs. % of Cycle
                velocity_sim = sim_results['sim_data']['velocity']
                velocity_theoretical = theoretical_results['sim_data']['velocity']
    
                axs[0, 1].plot(cycle_pct_sim, velocity_sim, color="green", label='Simulated Velocity')
                axs[0, 1].plot(cycle_pct_theoretical, velocity_theoretical, color="lightgreen", linestyle='--', label='Theoretical Velocity')
                axs[0, 1].set_title("Velocity vs. % of Cycle")
                axs[0, 1].set_xlabel("% of Cycle")
                axs[0, 1].set_ylabel("Velocity (m/s)")
                

                # Bottom-left: Position vs. % of Cycle
                position_sim = sim_results['sim_data']['position']
                position_theoretical = theoretical_results['sim_data']['position']
    
                axs[1, 0].plot(cycle_pct_sim, position_sim, color="orange", label='Simulated Position')
                axs[1, 0].plot(cycle_pct_theoretical, position_theoretical, color="darkorange", linestyle='--', label='Theoretical Position')
                axs[1, 0].set_title("Position vs. % of Cycle")
                axs[1, 0].set_xlabel("% of Cycle")
                axs[1, 0].set_ylabel("Position (m)")
                

                # Bottom-right: Power vs. % of Cycle
                power_sim = sim_results['sim_data']['power']
                power_theoretical = theoretical_results['sim_data']['power']
    
                axs[1, 1].plot(cycle_pct_sim, power_sim, color="red", label='Simulated Power')
                axs[1, 1].plot(cycle_pct_theoretical, power_theoretical, color="lightcoral", linestyle='--', label='Theoretical Power')
                axs[1, 1].set_title("Power vs. % of Cycle")
                axs[1, 1].set_xlabel("% of Cycle")
                axs[1, 1].set_ylabel("Power (W)")
                

                # Adjust layout
                fig.tight_layout()
                return fig


            @render.plot
            def work_loop1():
                results = run_simulation()
                sim_results = results[0]  # First item in the tuple is sim_results
                theoretical_results = results[1]  # Second item is theoretical_results
                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                fig, ax = plt.subplots()

                # Extract force and position data for the work-loop graph
                force_total_sim = sim_results['sim_data']['force_total']
                position_sim = sim_results['sim_data']['position']
    
                force_total_theoretical = theoretical_results['sim_data']['force_total']
                position_theoretical = theoretical_results['sim_data']['position']

                # Plot force vs. position (excursion)
                ax.plot(position_sim, force_total_sim, label="Simulated Work Loop")
                ax.plot(position_theoretical, force_total_theoretical, label="Theoretical Work Loop", linestyle='--')

                ax.set_title("Work Loop (Force vs. Excursion)")
                ax.set_xlabel("Excursion (m)")
                ax.set_ylabel("Force (N)")
                ax.legend()

                return fig

        with ui.nav_panel(title="Graphs2"):
            
# Define Server logic and plotting
   
            @render.plot
            def force_cycle_with_theoretical():
                results = run_simulation()
                sim_results = results[0]  # First item in the tuple is sim_results
                theoretical_results = results[1]  # Second item is theoretical_results

                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                # Create the plot
                fig, ax = plt.subplots()

                # Plot Simulated Force vs. % of Cycle
                ax.plot(sim_results['sim_data']['cycle_pct'], sim_results['sim_data']['force_total'], label='Simulated Force', color='blue')

                # Plot Theoretical Force vs. % of Cycle
                ax.plot(theoretical_results['sim_data']['cycle_pct'], theoretical_results['sim_data']['force_total'], label='Theoretical Force', color='orange')

                # Add labels and legend
                ax.set_title("Force vs. % of Cycle (Simulated vs. Theoretical)")
                ax.set_xlabel("% of Cycle")
                ax.set_ylabel("Force (N)")
                ax.legend()

                return fig


            @render.plot
            def velocity_cycle():
                results = run_simulation()
                sim_results = results[0]
                theoretical_results = results[1]

                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                # Create the plot
                fig, ax = plt.subplots()

                # Plot Simulated Velocity vs. % of Cycle
                ax.plot(sim_results['sim_data']['cycle_pct'], sim_results['sim_data']['velocity'], label='Simulated Velocity', color='green')

                # Plot Theoretical Velocity vs. % of Cycle
                ax.plot(theoretical_results['sim_data']['cycle_pct'], theoretical_results['sim_data']['velocity'], label='Theoretical Velocity', color='lightgreen')

                # Add labels and legend
                ax.set_title("Velocity vs. % of Cycle (Simulated vs. Theoretical)")
                ax.set_xlabel("% of Cycle")
                ax.set_ylabel("Velocity (m/s)")
                ax.legend()

                return fig


            @render.plot
            def position_cycle():
                results = run_simulation()
                sim_results = results[0]
                theoretical_results = results[1]

                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                # Create the plot
                fig, ax = plt.subplots()

                # Plot Simulated Position vs. % of Cycle
                ax.plot(sim_results['sim_data']['cycle_pct'], sim_results['sim_data']['position'], label='Simulated Position', color='orange')

                # Plot Theoretical Position vs. % of Cycle
                ax.plot(theoretical_results['sim_data']['cycle_pct'], theoretical_results['sim_data']['position'], label='Theoretical Position', color='darkorange')

                # Add labels and legend
                ax.set_title("Position vs. % of Cycle (Simulated vs. Theoretical)")
                ax.set_xlabel("% of Cycle")
                ax.set_ylabel("Position (m)")
                ax.legend()
    
                return fig


            @render.plot
            def power_cycle():
                results = run_simulation()
                sim_results = results[0]
                theoretical_results = results[1]

                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                # Create the plot
                fig, ax = plt.subplots()

                # Plot Simulated Power vs. % of Cycle
                ax.plot(sim_results['sim_data']['cycle_pct'], sim_results['sim_data']['power'], label='Simulated Power', color='red')

                # Plot Theoretical Power vs. % of Cycle
                ax.plot(theoretical_results['sim_data']['cycle_pct'], theoretical_results['sim_data']['power'], label='Theoretical Power', color='lightcoral')

                # Add labels and legend
                ax.set_title("Power vs. % of Cycle (Simulated vs. Theoretical)")
                ax.set_xlabel("% of Cycle")
                ax.set_ylabel("Power (W)")
                ax.legend()

                return fig


            @render.plot
            def work_loop():
                results = run_simulation()
                sim_results = results[0]
                theoretical_results = results[1]

                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                # Create the plot
                fig, ax = plt.subplots()

                # Extract force and position data for the work-loop graph
                force_total_sim = sim_results['sim_data']['force_total']
                position_sim = sim_results['sim_data']['position']
    
                force_total_theoretical = theoretical_results['sim_data']['force_total']
                position_theoretical = theoretical_results['sim_data']['position']

                # Plot force vs. position (excursion) for both simulated and theoretical
                ax.plot(position_sim, force_total_sim, label="Simulated Work Loop", color='blue')
                ax.plot(position_theoretical, force_total_theoretical, label="Theoretical Work Loop", linestyle='--', color='orange')

                # Add labels and legend
                ax.set_title("Work Loop (Force vs. Excursion)")
                ax.set_xlabel("Excursion (m)")
                ax.set_ylabel("Force (N)")
                ax.legend()

                return fig

   
@render.table
def results():
    # Unpack the tuple into sim_results and theoretical_results
    results = run_simulation()
    sim_results = results[0]  # First item in the tuple is sim_results
    theoretical_results = results[1]  # Second item is theoretical_results

    if sim_results is None or theoretical_results is None:
        print("Simulation failed: one or both result sets are None")
        return pd.DataFrame()

    # Create a DataFrame with both simulated and theoretical results
    results_df = pd.DataFrame({
        "Metric": ["Total Work", "Positive Work", "Negative Work", "Power"],
        "Simulated Results": [
            sim_results['work_actual'],
            sim_results['work_positive'],
            sim_results['work_negative'],
            sim_results['power_actual']
        ],
        "Theoretical Results": [
            theoretical_results['work_actual'],
            theoretical_results['work_positive'],
            theoretical_results['work_negative'],
            theoretical_results['power_actual']
        ]
    })

    return results_df
   # New function to render the optimized onset/offset table
@render.table
def optimized_onset_offset():
    # Show optimized onset/offset only if the optimize checkbox is selected
    if input.optimize():
        # Call run_simulation and assume it returns the optimized onset and offset as well
        optimized_results = thelen_muscle_opt(input.cycle_freq(), input.excursion(), input.length_optimal(),
                                              input.max_isometric_force(), input.max_velocity(),
                                              input.force_velocity_curvature(), input.activation_time(), input.deactivation_time())
        if optimized_results is None:
            return pd.DataFrame()

        # Extract optimized onset and offset values from the optimization function
        _, optimized_onset, optimized_offset, _ = optimized_results

        # Create a DataFrame for the optimized values
        optimization_df = pd.DataFrame({
            "Parameter": ["Optimized Onset", "Optimized Offset"],
            "Value": [optimized_onset, optimized_offset]
        })

        return optimization_df
    else:
        # If optimization is not selected, return an empty DataFrame
        return pd.DataFrame()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import reactive
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

# Optimization function for the Thelen model (two-phase coarse→fine search)
def thelen_muscle_opt(freq, excursion, L0, F0, Vx, af, tau_a, tau_d):
    best_onset = None
    best_offset = None
    max_power = -np.inf

    # Phase 1: coarse search (step=5) over full range
    for onset in range(0, 75, 5):
        for offset in range(onset + 5, 100, 5):
            sim_results = thelen_muscle([onset, offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
            if sim_results is None:
                continue
            current_power = sim_results['power_actual']
            if current_power > max_power:
                best_onset = onset
                best_offset = offset
                max_power = current_power

    if best_onset is None:
        return None, None, None, None

    # Phase 2: fine search (step=1) within ±5 of coarse best
    max_power = -np.inf
    fine_best_onset = best_onset
    fine_best_offset = best_offset
    for onset in range(max(0, best_onset - 5), min(75, best_onset + 6)):
        for offset in range(max(onset + 1, best_offset - 5), min(100, best_offset + 6)):
            sim_results = thelen_muscle([onset, offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
            if sim_results is None:
                continue
            current_power = sim_results['power_actual']
            if current_power > max_power:
                fine_best_onset = onset
                fine_best_offset = offset
                max_power = current_power

    optimized_results = thelen_muscle([fine_best_onset, fine_best_offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
    if optimized_results is not None:
        optimized_results['best_onset'] = fine_best_onset
        optimized_results['best_offset'] = fine_best_offset
    return optimized_results, fine_best_onset, fine_best_offset, max_power

# Define the run_simulation function with theoretical results
@reactive.calc
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
    
    # Always compute slider-based simulation
    sim_results = thelen_muscle(**muscle_params)

    # Compute optimized results only when checkbox is checked
    opt_results = None
    if input.optimize():
        opt_results, _, _, _ = thelen_muscle_opt(
            input.cycle_freq(), input.excursion(), input.length_optimal(),
            input.max_isometric_force(), input.max_velocity(),
            input.force_velocity_curvature(), input.activation_time(),
            input.deactivation_time()
        )

    # Theoretical muscle parameters (instantaneous activation/deactivation and no onoff)
    theoretical_params = muscle_params.copy()
    theoretical_params['onoff'] = [25, 75]  # No onset or offset
    theoretical_params['tau_a'] = 0.5  # No activation time
    theoretical_params['tau_d'] = 1  # No deactivation time
    
    # Calculate theoretical results with zero onset/offset and instantaneous activation/deactivation
    theoretical_results = thelen_muscle(**theoretical_params)
    
    return sim_results, theoretical_results, opt_results

# Info modal handler
@reactive.effect
@reactive.event(input.info_btn)
def show_info_modal():
    m = ui.modal(
        ui.p("Developed by Jim Martin, Jenna Link, Marc Klimstra"),
        title="About Virtual Muscle Lab",
        easy_close=True,
        footer=ui.modal_button("Close")
    )
    ui.modal_show(m)

# Define the UI layout
ui.tags.style("""
    body, .bslib-page-fill { padding-top: 52px !important; }
    .app-banner { position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
                  background-color: #2c5f8a; color: white;
                  padding: 10px 20px; display: flex; align-items: center; }
""")
ui.div(
    ui.span("Virtual Muscle Lab", style="font-size:1.5em; font-weight:bold; color:white;"),
    ui.input_action_button(
        "info_btn", "\u24d8",
        style="background:none; border:1px solid rgba(255,255,255,0.6); border-radius:50%; color:white; font-size:1em; width:30px; height:30px; line-height:1; cursor:pointer; margin-left:12px; padding:0;"
    ),
    class_="app-banner"
)

with ui.sidebar():
            ui.input_slider("onset", "Onset (% of cycle)", min=0, max=74, value=22),
            ui.input_slider("offset", "Offset (% of cycle)", min=1, max=99, value=66),
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
    with ui.navset_bar(title=""):
        with ui.nav_panel(title="Force,Velocity,Power"):
            ui.input_switch("show_workloop", "Show Workloop", value=False)
            @render.plot
            def combined_graphs():
                results = run_simulation()
                sim_results = results[0]
                theoretical_results = results[1]
                opt_results = results[2]
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
                if opt_results is not None:
                    axs[0, 0].plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['force_total'], label='Optimized Force', linestyle=':', color='purple')
                axs[0, 0].set_title("Force vs. % of Cycle")
                axs[0, 0].set_xlabel("% of Cycle")
                axs[0, 0].set_ylabel("Force (N)")
                axs[0, 0].legend()

                # Top-right: Velocity vs. % of Cycle
                velocity_sim = sim_results['sim_data']['velocity']
                velocity_theoretical = theoretical_results['sim_data']['velocity']
    
                axs[0, 1].plot(cycle_pct_sim, velocity_sim, color="green", label='Simulated Velocity')
                axs[0, 1].plot(cycle_pct_theoretical, velocity_theoretical, color="lightgreen", linestyle='--', label='Theoretical Velocity')
                if opt_results is not None:
                    axs[0, 1].plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['velocity'], label='Optimized Velocity', linestyle=':', color='purple')
                axs[0, 1].set_title("Velocity vs. % of Cycle")
                axs[0, 1].set_xlabel("% of Cycle")
                axs[0, 1].set_ylabel("Velocity (m/s)")
                axs[0, 1].legend()

                # Bottom-left: Position vs. % of Cycle
                position_sim = sim_results['sim_data']['position']
                position_theoretical = theoretical_results['sim_data']['position']
    
                axs[1, 0].plot(cycle_pct_sim, position_sim, color="orange", label='Simulated Position')
                axs[1, 0].plot(cycle_pct_theoretical, position_theoretical, color="darkorange", linestyle='--', label='Theoretical Position')
                if opt_results is not None:
                    axs[1, 0].plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['position'], label='Optimized Position', linestyle=':', color='purple')
                axs[1, 0].set_title("Position vs. % of Cycle")
                axs[1, 0].set_xlabel("% of Cycle")
                axs[1, 0].set_ylabel("Position (m)")
                axs[1, 0].legend()

                # Bottom-right: Power vs. % of Cycle
                power_sim = sim_results['sim_data']['power']
                power_theoretical = theoretical_results['sim_data']['power']
    
                axs[1, 1].plot(cycle_pct_sim, power_sim, color="red", label='Simulated Power')
                axs[1, 1].plot(cycle_pct_theoretical, power_theoretical, color="lightcoral", linestyle='--', label='Theoretical Power')
                if opt_results is not None:
                    axs[1, 1].plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['power'], label='Optimized Power', linestyle=':', color='purple')
                axs[1, 1].set_title("Power vs. % of Cycle")
                axs[1, 1].set_xlabel("% of Cycle")
                axs[1, 1].set_ylabel("Power (W)")
                axs[1, 1].legend()

                # Adjust layout
                fig.tight_layout()
                return fig

            with ui.panel_conditional("input.show_workloop"):

                @render.plot
                def work_loop1():
                    results = run_simulation()
                    sim_results = results[0]
                    theoretical_results = results[1]
                    opt_results = results[2]
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
                    if opt_results is not None:
                        ax.plot(opt_results['sim_data']['position'], opt_results['sim_data']['force_total'], label="Optimized Work Loop", linestyle=':', color='purple')

                    ax.set_title("Work Loop (Force vs. Excursion)")
                    ax.set_xlabel("Excursion (m)")
                    ax.set_ylabel("Force (N)")
                    ax.legend()

                    return fig

                @render.ui
                def workloop_metrics():
                    r = run_simulation()
                    sim = r[0]
                    theo = r[1]
                    opt = r[2]
                    if sim is None or theo is None:
                        return ui.p("No results available")
                    opt_col = [
                        round(opt['work_actual'], 4),
                        round(opt['work_positive'], 4),
                        round(opt['work_negative'], 4),
                        round(opt['power_actual'], 4),
                    ] if opt is not None else ["\u2014", "\u2014", "\u2014", "\u2014"]
                    df = pd.DataFrame({
                        "Metric": ["Total Work (J)", "Positive Work (J)", "Negative Work (J)", "Mean Power (W)"],
                        "Simulated": [round(sim['work_actual'], 4), round(sim['work_positive'], 4), round(sim['work_negative'], 4), round(sim['power_actual'], 4)],
                        "Theoretical": [round(theo['work_actual'], 4), round(theo['work_positive'], 4), round(theo['work_negative'], 4), round(theo['power_actual'], 4)],
                        "Optimized": opt_col,
                    })
                    return ui.HTML(df.to_html(index=False, classes="table table-bordered table-sm text-center", border=0))

        with ui.nav_panel(title="Graphs2"):
            
# Define Server logic and plotting
   
            @render.plot
            def force_cycle_with_theoretical():
                results = run_simulation()
                sim_results = results[0]
                theoretical_results = results[1]
                opt_results = results[2]

                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                # Create the plot
                fig, ax = plt.subplots()

                # Plot Simulated Force vs. % of Cycle
                ax.plot(sim_results['sim_data']['cycle_pct'], sim_results['sim_data']['force_total'], label='Simulated Force', color='blue')

                # Plot Theoretical Force vs. % of Cycle
                ax.plot(theoretical_results['sim_data']['cycle_pct'], theoretical_results['sim_data']['force_total'], label='Theoretical Force', color='orange')

                if opt_results is not None:
                    ax.plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['force_total'], label='Optimized Force', linestyle=':', color='purple')

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
                opt_results = results[2]

                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                # Create the plot
                fig, ax = plt.subplots()

                # Plot Simulated Velocity vs. % of Cycle
                ax.plot(sim_results['sim_data']['cycle_pct'], sim_results['sim_data']['velocity'], label='Simulated Velocity', color='green')

                # Plot Theoretical Velocity vs. % of Cycle
                ax.plot(theoretical_results['sim_data']['cycle_pct'], theoretical_results['sim_data']['velocity'], label='Theoretical Velocity', color='lightgreen')

                if opt_results is not None:
                    ax.plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['velocity'], label='Optimized Velocity', linestyle=':', color='purple')

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
                opt_results = results[2]

                if sim_results is None or theoretical_results is None:
                    print("Simulation failed: one or both result sets are None")
                    return

                # Create the plot
                fig, ax = plt.subplots()

                # Plot Simulated Power vs. % of Cycle
                ax.plot(sim_results['sim_data']['cycle_pct'], sim_results['sim_data']['power'], label='Simulated Power', color='red')

                # Plot Theoretical Power vs. % of Cycle
                ax.plot(theoretical_results['sim_data']['cycle_pct'], theoretical_results['sim_data']['power'], label='Theoretical Power', color='lightcoral')

                if opt_results is not None:
                    ax.plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['power'], label='Optimized Power', linestyle=':', color='purple')

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
                opt_results = results[2]

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
                if opt_results is not None:
                    ax.plot(opt_results['sim_data']['position'], opt_results['sim_data']['force_total'], label="Optimized Work Loop", linestyle=':', color='purple')

                # Add labels and legend
                ax.set_title("Work Loop (Force vs. Excursion)")
                ax.set_xlabel("Excursion (m)")
                ax.set_ylabel("Force (N)")
                ax.legend()

                return fig

   # New function to render the optimized onset/offset table
@render.table
def optimized_onset_offset():
    if input.optimize():
        results = run_simulation()
        opt_results = results[2]
        if opt_results is None:
            return pd.DataFrame()
        optimization_df = pd.DataFrame({
            "Parameter": ["Optimized Onset", "Optimized Offset"],
            "Value": [opt_results['best_onset'], opt_results['best_offset']]
        })
        return optimization_df
    else:
        return pd.DataFrame()

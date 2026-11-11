/**
 * Shared wire fixtures for the OPTIMIZER panel's Vitest files: the service's
 * own JSON shapes (EnvironmentDetails, a resolved OptimizationConfig, a
 * detail=full JobResponse), so every test reads the field names the service
 * actually sends.
 */

/** `GET environments/{name}` */
export function envDetails() {
  return {
    name: 'injector',
    environment_type: 'simulation',
    variables: {
      phase_a: {
        min_value: -180,
        max_value: 180,
        units: 'deg',
        description: 'Buncher phase',
        readback_pv: 'A:RB',
        setpoint_pv: 'A:SP',
        step_size: 0.5,
        max_step: 20,
      },
      sol_b: {
        min_value: 0,
        max_value: 10,
        units: 'A',
        description: '',
        readback_pv: 'B:RB',
        setpoint_pv: 'B:SP',
        step_size: 0.01,
        max_step: null,
      },
    },
    observables: {
      charge: { kind: 'pv', readback_pv: 'Q', display_name: 'Charge', direction: 'MAXIMIZE' },
      loss: { kind: 'pv', readback_pv: 'L', direction: 'MINIMIZE' },
      plain: { kind: 'pv', readback_pv: 'P' },
    },
    default_objective: 'charge',
    available_objectives: ['charge', 'loss', 'plain'],
  };
}

/** A stored, resolved draft (`GET jobs/{id}/config`). */
export function storedConfig(overrides = {}) {
  return {
    environment_name: 'injector',
    environment_config: {},
    preset: null,
    control_target: 'va',
    execution_mode: null,
    on_complete: 'apply_best',
    objective_name: 'charge',
    objective_direction: null,
    algorithm: 'upper_confidence_bound',
    n_iterations: 30,
    n_initial_samples: null,
    variables: ['phase_a', 'sol_b'],
    variable_overrides: {
      phase_a: { min: -18, max: 18, step_size: 0.5, window_fraction: null, window_abs: null },
      sol_b: { min: 2, max: 4, step_size: 0.01, window_fraction: null, window_abs: null },
    },
    surrogate_model: 'default',
    model_file: null,
    algorithm_params: null,
    constraints: {},
    ...overrides,
  };
}

/** `GET jobs/{id}?detail=full` for a run in progress. */
export function runningJob(overrides = {}) {
  return {
    job_id: 'job-1234-abcd',
    status: 'running',
    error_message: null,
    data_version: 3,
    environment_name: 'injector',
    objective_name: 'charge',
    objective_direction: 'MAXIMIZE',
    variable_names: ['phase_a', 'sol_b'],
    total_evaluations: 10,
    data: [
      { phase_a: 0, sol_b: 3, charge: 1.0 },
      { phase_a: 9, sol_b: 2, charge: 2.5 },
      { phase_a: -9, sol_b: 4, charge: 2.0 },
    ],
    variable_bounds: { phase_a: [-18, 18], sol_b: [2, 4] },
    write_authorization: null,
    job_type: 'optimization',
    ...overrides,
  };
}

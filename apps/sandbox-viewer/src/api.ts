import type {
  CreateSimulationResponse,
  InstructionResponse,
  SimulationAction,
  StepResponse,
  WorldState,
} from './types';

const DEFAULT_BASE_URL = '/api';

export class SimulationApi {
  private readonly baseUrl: string;

  constructor(baseUrl = import.meta.env.VITE_SIM_API_BASE_URL || DEFAULT_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async createSimulation(simulationId: string): Promise<CreateSimulationResponse> {
    const response = await this.request<CreateSimulationResponse>('/simulations', {
      method: 'POST',
      body: JSON.stringify({ simulation_id: simulationId }),
    });
    return response;
  }

  async getState(simulationId: string): Promise<WorldState> {
    return this.request<WorldState>(`/simulations/${simulationId}/state`);
  }

  async step(simulationId: string, ticks = 1, actions: SimulationAction[] = []): Promise<StepResponse> {
    return this.request<StepResponse>(`/simulations/${simulationId}/step`, {
      method: 'POST',
      body: JSON.stringify({ ticks, actions }),
    });
  }

  async sendInstruction(simulationId: string, instruction: string): Promise<InstructionResponse> {
    return this.request<InstructionResponse>(`/simulations/${simulationId}/instructions`, {
      method: 'POST',
      body: JSON.stringify({ instruction }),
    });
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init.headers || {}),
      },
    });

    if (!response.ok) {
      const detail = await readErrorDetail(response);
      throw new Error(detail || `${response.status} ${response.statusText}`);
    }

    return response.json() as Promise<T>;
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === 'string') {
      return payload.detail;
    }
    return JSON.stringify(payload.detail ?? payload);
  } catch {
    return response.statusText;
  }
}

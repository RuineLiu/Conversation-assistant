export type Tile = [number, number];

export type ObjectType =
  | 'door'
  | 'conference_table'
  | 'chair'
  | 'screen'
  | 'whiteboard'
  | 'light_switch'
  | 'laptop'
  | 'documents'
  | 'cup'
  | 'remote';

export interface ConferenceRoomMap {
  width: number;
  height: number;
  tile_size: number;
  meters_per_tile: number;
  layers: string[];
  wall_tiles: Tile[];
  spawn_points: Record<string, Tile>;
}

export interface SimulationObject {
  object_id: string;
  type: ObjectType;
  label: string;
  tile: Tile;
  size: Tile;
  blocks_movement: boolean;
  interaction_tiles: Tile[];
  state: Record<string, unknown>;
}

export interface AssistantState {
  wearing_device: boolean;
  last_intervention_tick: number | null;
  interruption_load: number;
}

export interface AgentState {
  agent_id: string;
  persona_id: string;
  display_name: string;
  role: 'presenter' | 'participant' | 'facilitator';
  tile: Tile;
  facing: 'north' | 'east' | 'south' | 'west';
  current_goal: string;
  current_action: string | null;
  path: Tile[];
  public_mood: 'focused' | 'neutral' | 'confused' | 'interrupted';
  assistant_state: AssistantState;
}

export interface DialogueBubble {
  agent_id: string;
  utterance: string;
  audience: string[];
  started_tick: number;
}

export interface AssistantInterventionAction {
  type: 'assistant_intervention';
  intervention_id: string;
  target_agent_id: string;
  timing_policy: 'before_activity' | 'during_activity' | 'after_activity';
  channel: 'visual' | 'audio' | 'haptic';
  content: string;
  rationale: string;
  metadata: Record<string, unknown>;
}

export interface SimulationEvent {
  event_id: string;
  tick: number;
  event_type: string;
  actor_id: string | null;
  target_id: string | null;
  payload: Record<string, unknown>;
}

export interface WorldState {
  simulation_id: string;
  tick: number;
  scenario_id: string;
  map: ConferenceRoomMap;
  agents: Record<string, AgentState>;
  objects: Record<string, SimulationObject>;
  active_dialogue: DialogueBubble[];
  active_interventions: AssistantInterventionAction[];
  transcript_events: unknown[];
  recent_events: SimulationEvent[];
  event_log: SimulationEvent[];
  metadata: Record<string, unknown>;
}

export type SimulationAction =
  | {
      type: 'move_to';
      agent_id: string;
      target_tile: Tile;
      reason: string;
    }
  | {
      type: 'speak';
      agent_id: string;
      utterance: string;
      audience: string[];
    }
  | {
      type: 'update_goal';
      agent_id: string;
      goal: string;
    }
  | AssistantInterventionAction;

export interface CreateSimulationResponse {
  simulation_id: string;
  state: WorldState;
}

export interface StepResponse {
  state: WorldState;
  events: SimulationEvent[];
}

export interface InstructionResponse {
  state: WorldState;
  event: SimulationEvent;
}

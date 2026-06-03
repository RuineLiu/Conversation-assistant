import Phaser from 'phaser';
import { SimulationApi } from './api';
import { SandboxScene } from './SandboxScene';
import type { SimulationAction, SimulationEvent, WorldState } from './types';
import './styles.css';

const api = new SimulationApi();
const simulationId = window.localStorage.getItem('sandbox-simulation-id') || 'sandbox_viewer_demo';

const game = new Phaser.Game({
  type: Phaser.AUTO,
  parent: 'game-root',
  backgroundColor: '#e6e1d7',
  scale: {
    mode: Phaser.Scale.RESIZE,
    autoCenter: Phaser.Scale.CENTER_BOTH,
    width: '100%',
    height: '100%',
  },
  render: {
    antialias: true,
    pixelArt: false,
  },
  scene: [SandboxScene],
});

void game;

const elements = {
  status: requireElement<HTMLDivElement>('connection-status'),
  tickCounter: requireElement<HTMLSpanElement>('tick-counter'),
  agentCount: requireElement<HTMLSpanElement>('agent-count'),
  objectCount: requireElement<HTMLSpanElement>('object-count'),
  eventCount: requireElement<HTMLSpanElement>('event-count'),
  agentList: requireElement<HTMLDivElement>('agent-list'),
  objectList: requireElement<HTMLDivElement>('object-list'),
  eventList: requireElement<HTMLDivElement>('event-list'),
  stepOnce: requireElement<HTMLButtonElement>('step-once'),
  stepFive: requireElement<HTMLButtonElement>('step-five'),
  demoActions: requireElement<HTMLButtonElement>('demo-actions'),
  instructionForm: requireElement<HTMLFormElement>('instruction-form'),
  instructionInput: requireElement<HTMLInputElement>('instruction-input'),
};

let currentState: WorldState | null = null;
let busy = false;

bootstrap();

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`missing element: ${id}`);
  }
  return element as T;
}

async function bootstrap(): Promise<void> {
  bindControls();
  setStatus('Connecting', 'pending');
  try {
    const created = await api.createSimulation(simulationId);
    window.localStorage.setItem('sandbox-simulation-id', created.simulation_id);
    applyState(created.state);
    setStatus('Connected', 'ok');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message.includes('simulation already exists')) {
      const state = await api.getState(simulationId);
      applyState(state);
      setStatus('Connected', 'ok');
      return;
    }
    setStatus('API offline', 'error');
    renderError(message);
  }
}

function bindControls(): void {
  elements.stepOnce.addEventListener('click', () => {
    void runStep(1);
  });
  elements.stepFive.addEventListener('click', () => {
    void runStep(5);
  });
  elements.demoActions.addEventListener('click', () => {
    void runDemoActions();
  });
  elements.instructionForm.addEventListener('submit', (event) => {
    event.preventDefault();
    void sendInstruction();
  });
}

async function runStep(ticks: number): Promise<void> {
  if (!currentState || busy) {
    return;
  }
  await withBusy(async () => {
    const response = await api.step(currentState!.simulation_id, ticks);
    applyState(response.state);
  });
}

async function runDemoActions(): Promise<void> {
  if (!currentState || busy) {
    return;
  }
  const actions: SimulationAction[] = [
    {
      type: 'move_to',
      agent_id: 'agent_alex',
      target_tile: [11, 6],
      reason: 'prepare presenter laptop',
    },
    {
      type: 'move_to',
      agent_id: 'agent_bao',
      target_tile: [15, 6],
      reason: 'review shared documents',
    },
    {
      type: 'move_to',
      agent_id: 'agent_chris',
      target_tile: [9, 2],
      reason: 'stand near whiteboard',
    },
    {
      type: 'speak',
      agent_id: 'agent_alex',
      utterance: 'Let us get the screen and notes ready before the meeting starts.',
      audience: ['agent_bao', 'agent_chris'],
    },
  ];

  await withBusy(async () => {
    const response = await api.step(currentState!.simulation_id, 1, actions);
    applyState(response.state);
  });
}

async function sendInstruction(): Promise<void> {
  if (!currentState || busy) {
    return;
  }
  const instruction = elements.instructionInput.value.trim();
  if (!instruction) {
    return;
  }
  await withBusy(async () => {
    const response = await api.sendInstruction(currentState!.simulation_id, instruction);
    elements.instructionInput.value = '';
    applyState(response.state);
  });
}

async function withBusy(task: () => Promise<void>): Promise<void> {
  busy = true;
  setControlsDisabled(true);
  try {
    await task();
    setStatus('Connected', 'ok');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus('Action failed', 'error');
    renderError(message);
  } finally {
    busy = false;
    setControlsDisabled(false);
  }
}

function setControlsDisabled(disabled: boolean): void {
  elements.stepOnce.disabled = disabled;
  elements.stepFive.disabled = disabled;
  elements.demoActions.disabled = disabled;
  elements.instructionInput.disabled = disabled;
  elements.instructionForm.querySelector('button')?.toggleAttribute('disabled', disabled);
}

function applyState(state: WorldState): void {
  currentState = state;
  window.dispatchEvent(new CustomEvent('sandbox:world-state', { detail: state }));
  renderPanel(state);
}

function renderPanel(state: WorldState): void {
  elements.tickCounter.textContent = `Tick ${state.tick}`;
  const agents = Object.values(state.agents);
  const objects = Object.values(state.objects);
  const events = state.event_log.slice(-12).reverse();

  elements.agentCount.textContent = String(agents.length);
  elements.objectCount.textContent = String(objects.length);
  elements.eventCount.textContent = String(state.event_log.length);

  elements.agentList.replaceChildren(...agents.map(renderAgentCard));
  elements.objectList.replaceChildren(...objects.filter((object) => object.type !== 'chair').map(renderObjectCard));
  elements.eventList.replaceChildren(...events.map(renderEventItem));
}

function renderAgentCard(agent: WorldState['agents'][string]): HTMLElement {
  const card = document.createElement('article');
  card.className = 'agent-card';

  card.append(
    renderTopRow(agent.display_name, agent.role),
    renderParagraph(agent.current_goal || 'No goal'),
    renderAgentMeta([
      agent.public_mood,
      agent.current_action || 'idle',
      `[${agent.tile.join(', ')}]`,
    ]),
  );
  return card;
}

function renderObjectCard(object: WorldState['objects'][string]): HTMLElement {
  const card = document.createElement('article');
  card.className = 'object-card';
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(object.state, null, 2);
  card.append(renderTopRow(object.label, object.type), pre);
  return card;
}

function renderEventItem(event: SimulationEvent): HTMLElement {
  const item = document.createElement('article');
  item.className = 'event-item';
  item.append(
    renderTopRow(event.event_type, `tick ${event.tick}`),
    renderParagraph(`${event.actor_id || 'system'}${event.target_id ? ` -> ${event.target_id}` : ''}`),
  );
  return item;
}

function renderTopRow(primaryText: string, secondaryText: string): HTMLElement {
  const row = document.createElement('div');
  row.className = 'agent-card-top';
  const primary = document.createElement('strong');
  primary.textContent = primaryText;
  const secondary = document.createElement('span');
  secondary.textContent = secondaryText;
  row.append(primary, secondary);
  return row;
}

function renderParagraph(text: string): HTMLParagraphElement {
  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  return paragraph;
}

function renderAgentMeta(values: string[]): HTMLElement {
  const meta = document.createElement('div');
  meta.className = 'agent-meta';
  for (const value of values) {
    const item = document.createElement('span');
    item.textContent = value;
    meta.append(item);
  }
  return meta;
}

function renderError(message: string): void {
  elements.eventList.replaceChildren();
  const item = document.createElement('article');
  item.className = 'event-item error-item';
  const title = document.createElement('div');
  const strong = document.createElement('strong');
  strong.textContent = 'Frontend cannot reach the simulation API';
  title.append(strong);
  item.append(title, renderParagraph(message));
  elements.eventList.append(item);
}

function setStatus(label: string, tone: 'pending' | 'ok' | 'error'): void {
  elements.status.textContent = label;
  elements.status.dataset.tone = tone;
}

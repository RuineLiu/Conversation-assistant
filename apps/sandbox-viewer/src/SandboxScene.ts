import Phaser from 'phaser';
import type { AgentState, SimulationObject, Tile, WorldState } from './types';

const TILE_SIZE = 32;
const FLOOR_COLOR = 0xd8c7a4;
const GRID_COLOR = 0xb9aa8b;
const WALL_COLOR = 0x3b3d3f;
const INTERACTION_COLOR = 0x86c5bf;

interface RenderConfig {
  mapOffsetX: number;
  mapOffsetY: number;
  zoom: number;
}

export class SandboxScene extends Phaser.Scene {
  private worldState: WorldState | null = null;
  private rootLayer?: Phaser.GameObjects.Container;
  private renderConfig: RenderConfig = { mapOffsetX: 0, mapOffsetY: 0, zoom: 1 };

  constructor() {
    super('SandboxScene');
  }

  create(): void {
    this.cameras.main.setBackgroundColor('#e6e1d7');
    window.addEventListener('sandbox:world-state', this.handleWorldState);
    this.scale.on('resize', this.handleResize, this);
  }

  shutdown(): void {
    window.removeEventListener('sandbox:world-state', this.handleWorldState);
    this.scale.off('resize', this.handleResize, this);
  }

  private readonly handleWorldState = (event: Event): void => {
    const customEvent = event as CustomEvent<WorldState>;
    this.worldState = customEvent.detail;
    this.renderWorld();
  };

  private handleResize(): void {
    this.renderWorld();
  }

  private renderWorld(): void {
    if (!this.worldState) {
      return;
    }

    this.rootLayer?.destroy(true);
    this.rootLayer = this.add.container(0, 0);
    this.renderConfig = this.calculateRenderConfig(this.worldState);

    this.drawFloor(this.worldState);
    this.drawInteractionZones(this.worldState);
    this.drawObjects(this.worldState);
    this.drawWalls(this.worldState);
    this.drawAgents(this.worldState);
    this.drawLegend(this.worldState);
  }

  private calculateRenderConfig(state: WorldState): RenderConfig {
    const canvasWidth = this.scale.width;
    const canvasHeight = this.scale.height;
    const mapPixelWidth = state.map.width * TILE_SIZE;
    const mapPixelHeight = state.map.height * TILE_SIZE;
    const zoom = Math.min(
      (canvasWidth - 48) / mapPixelWidth,
      (canvasHeight - 48) / mapPixelHeight,
      1.45,
    );
    return {
      zoom,
      mapOffsetX: (canvasWidth - mapPixelWidth * zoom) / 2,
      mapOffsetY: (canvasHeight - mapPixelHeight * zoom) / 2,
    };
  }

  private toScreen(tile: Tile): { x: number; y: number } {
    return {
      x: this.renderConfig.mapOffsetX + tile[0] * TILE_SIZE * this.renderConfig.zoom,
      y: this.renderConfig.mapOffsetY + tile[1] * TILE_SIZE * this.renderConfig.zoom,
    };
  }

  private tileSize(): number {
    return TILE_SIZE * this.renderConfig.zoom;
  }

  private drawFloor(state: WorldState): void {
    const graphics = this.add.graphics();
    const size = this.tileSize();
    for (let y = 0; y < state.map.height; y += 1) {
      for (let x = 0; x < state.map.width; x += 1) {
        const point = this.toScreen([x, y]);
        graphics.fillStyle(FLOOR_COLOR, 1);
        graphics.fillRect(point.x, point.y, size, size);
        graphics.lineStyle(1, GRID_COLOR, 0.38);
        graphics.strokeRect(point.x, point.y, size, size);
      }
    }
    this.rootLayer?.add(graphics);
  }

  private drawWalls(state: WorldState): void {
    const graphics = this.add.graphics();
    const size = this.tileSize();
    for (const tile of state.map.wall_tiles) {
      const point = this.toScreen(tile);
      graphics.fillStyle(WALL_COLOR, 1);
      graphics.fillRect(point.x, point.y, size, size);
    }
    this.rootLayer?.add(graphics);
  }

  private drawInteractionZones(state: WorldState): void {
    const graphics = this.add.graphics();
    const size = this.tileSize();
    for (const object of Object.values(state.objects)) {
      for (const tile of object.interaction_tiles) {
        const point = this.toScreen(tile);
        graphics.fillStyle(INTERACTION_COLOR, 0.18);
        graphics.fillRect(point.x + size * 0.12, point.y + size * 0.12, size * 0.76, size * 0.76);
      }
    }
    this.rootLayer?.add(graphics);
  }

  private drawObjects(state: WorldState): void {
    const sortedObjects = Object.values(state.objects).sort((a, b) => objectSortWeight(a) - objectSortWeight(b));
    for (const object of sortedObjects) {
      this.drawObject(object);
    }
  }

  private drawObject(object: SimulationObject): void {
    const graphics = this.add.graphics();
    const size = this.tileSize();
    const point = this.toScreen(object.tile);
    const width = object.size[0] * size;
    const height = object.size[1] * size;

    switch (object.type) {
      case 'conference_table':
        graphics.fillStyle(0xf6f1e8, 1);
        graphics.lineStyle(2, 0xa58f70, 0.9);
        graphics.fillRoundedRect(point.x, point.y, width, height, size * 0.8);
        graphics.strokeRoundedRect(point.x, point.y, width, height, size * 0.8);
        graphics.fillStyle(FLOOR_COLOR, 0.9);
        graphics.fillRoundedRect(point.x + width * 0.18, point.y + height * 0.22, width * 0.64, height * 0.56, size * 0.52);
        break;
      case 'chair':
        graphics.fillStyle(0x4d5960, 1);
        graphics.fillRoundedRect(point.x + size * 0.12, point.y + size * 0.16, size * 0.76, size * 0.58, size * 0.14);
        graphics.fillStyle(0x20272b, 1);
        graphics.fillRoundedRect(point.x + size * 0.18, point.y + size * 0.08, size * 0.64, size * 0.16, size * 0.08);
        graphics.fillCircle(point.x + size * 0.28, point.y + size * 0.82, size * 0.06);
        graphics.fillCircle(point.x + size * 0.72, point.y + size * 0.82, size * 0.06);
        break;
      case 'screen':
        graphics.fillStyle(0x101418, 1);
        graphics.fillRoundedRect(point.x + size * 0.12, point.y + size * 0.08, width * 0.76, height * 0.84, size * 0.08);
        graphics.lineStyle(2, 0x636a70, 0.9);
        graphics.strokeRoundedRect(point.x + size * 0.12, point.y + size * 0.08, width * 0.76, height * 0.84, size * 0.08);
        break;
      case 'whiteboard':
        graphics.fillStyle(0xf8f8f3, 1);
        graphics.lineStyle(2, 0x6a7f7d, 1);
        graphics.fillRoundedRect(point.x + size * 0.06, point.y + size * 0.12, width * 0.88, height * 0.72, size * 0.06);
        graphics.strokeRoundedRect(point.x + size * 0.06, point.y + size * 0.12, width * 0.88, height * 0.72, size * 0.06);
        break;
      case 'door':
        graphics.fillStyle(0x9fd1df, 0.42);
        graphics.lineStyle(2, 0x5e9daf, 0.9);
        graphics.fillRect(point.x + size * 0.08, point.y, width * 0.84, height);
        graphics.strokeRect(point.x + size * 0.08, point.y, width * 0.84, height);
        break;
      case 'light_switch':
        graphics.fillStyle(0xf4e3a1, 1);
        graphics.fillRoundedRect(point.x + size * 0.32, point.y + size * 0.2, size * 0.36, size * 0.6, size * 0.08);
        graphics.fillStyle(0x93845f, 1);
        graphics.fillCircle(point.x + size * 0.5, point.y + size * 0.5, size * 0.05);
        break;
      case 'laptop':
        graphics.fillStyle(0x25313a, 1);
        graphics.fillRoundedRect(point.x + size * 0.16, point.y + size * 0.24, size * 0.68, size * 0.4, size * 0.06);
        graphics.fillStyle(0x78a9c4, 1);
        graphics.fillRect(point.x + size * 0.24, point.y + size * 0.3, size * 0.52, size * 0.22);
        break;
      case 'documents':
        graphics.fillStyle(0xffffff, 1);
        graphics.lineStyle(1, 0xbababa, 1);
        graphics.fillRect(point.x + size * 0.18, point.y + size * 0.14, size * 0.54, size * 0.68);
        graphics.strokeRect(point.x + size * 0.18, point.y + size * 0.14, size * 0.54, size * 0.68);
        graphics.lineStyle(1, 0x8c8c8c, 0.7);
        graphics.lineBetween(point.x + size * 0.28, point.y + size * 0.34, point.x + size * 0.62, point.y + size * 0.34);
        graphics.lineBetween(point.x + size * 0.28, point.y + size * 0.48, point.x + size * 0.62, point.y + size * 0.48);
        break;
      case 'cup':
        graphics.fillStyle(0xbb6d47, 1);
        graphics.fillCircle(point.x + size * 0.48, point.y + size * 0.48, size * 0.22);
        graphics.fillStyle(0xf2d4b3, 1);
        graphics.fillCircle(point.x + size * 0.48, point.y + size * 0.48, size * 0.12);
        break;
      case 'remote':
        graphics.fillStyle(0x202428, 1);
        graphics.fillRoundedRect(point.x + size * 0.34, point.y + size * 0.16, size * 0.28, size * 0.68, size * 0.1);
        graphics.fillStyle(0x8fcf8d, 1);
        graphics.fillCircle(point.x + size * 0.48, point.y + size * 0.32, size * 0.05);
        break;
      default:
        graphics.fillStyle(0x777777, 1);
        graphics.fillRect(point.x, point.y, width, height);
    }

    this.rootLayer?.add(graphics);
    if (object.type !== 'chair' && object.type !== 'conference_table') {
      this.addLabel(shortObjectLabel(object), point.x + width / 2, point.y + height + size * 0.12, 11);
    }
  }

  private drawAgents(state: WorldState): void {
    const pathGraphics = this.add.graphics();
    pathGraphics.lineStyle(3, 0x2a6fbb, 0.58);
    for (const agent of Object.values(state.agents)) {
      this.drawAgentPath(agent, pathGraphics);
    }
    this.rootLayer?.add(pathGraphics);

    for (const agent of Object.values(state.agents)) {
      this.drawAgent(agent, state);
    }
  }

  private drawAgentPath(agent: AgentState, graphics: Phaser.GameObjects.Graphics): void {
    if (agent.path.length === 0) {
      return;
    }
    const size = this.tileSize();
    const start = this.tileCenter(agent.tile);
    graphics.beginPath();
    graphics.moveTo(start.x, start.y);
    for (const tile of agent.path) {
      const point = this.tileCenter(tile);
      graphics.lineTo(point.x, point.y);
    }
    graphics.strokePath();
    const target = this.tileCenter(agent.path[agent.path.length - 1]);
    graphics.fillStyle(0x2a6fbb, 0.86);
    graphics.fillCircle(target.x, target.y, size * 0.12);
  }

  private drawAgent(agent: AgentState, state: WorldState): void {
    const size = this.tileSize();
    const point = this.tileCenter(agent.tile);
    const color = agentColor(agent);
    const graphics = this.add.graphics();
    graphics.fillStyle(0xffffff, 1);
    graphics.fillCircle(point.x, point.y, size * 0.46);
    graphics.fillStyle(color, 1);
    graphics.fillCircle(point.x, point.y, size * 0.38);
    graphics.lineStyle(2, 0xffffff, 0.9);
    graphics.strokeCircle(point.x, point.y, size * 0.38);
    this.rootLayer?.add(graphics);

    this.addLabel(agent.display_name.slice(0, 1), point.x, point.y - size * 0.16, 15, '#ffffff');
    this.drawFacing(agent, point.x, point.y, size);

    const bubble = state.active_dialogue.find((item) => item.agent_id === agent.agent_id);
    if (bubble) {
      this.drawDialogueBubble(bubble.utterance, point.x, point.y - size * 0.82);
    }
  }

  private drawFacing(agent: AgentState, x: number, y: number, size: number): void {
    const graphics = this.add.graphics();
    graphics.fillStyle(0xffffff, 1);
    const offset = size * 0.28;
    const points: Record<AgentState['facing'], Tile> = {
      north: [0, -1],
      east: [1, 0],
      south: [0, 1],
      west: [-1, 0],
    };
    const delta = points[agent.facing];
    graphics.fillCircle(x + delta[0] * offset, y + delta[1] * offset, size * 0.065);
    this.rootLayer?.add(graphics);
  }

  private drawDialogueBubble(text: string, x: number, y: number): void {
    const trimmed = text.length > 38 ? `${text.slice(0, 35)}...` : text;
    const fontSize = Math.max(11, Math.round(13 * this.renderConfig.zoom));
    const label = this.add.text(x, y, trimmed, {
      color: '#152021',
      fontFamily: 'Inter, Arial, sans-serif',
      fontSize: `${fontSize}px`,
      align: 'center',
      wordWrap: { width: 180 * this.renderConfig.zoom },
    });
    label.setOrigin(0.5, 1);
    const bounds = label.getBounds();
    const graphics = this.add.graphics();
    graphics.fillStyle(0xffffff, 0.94);
    graphics.lineStyle(1, 0x9fb8b4, 0.9);
    graphics.fillRoundedRect(bounds.x - 8, bounds.y - 6, bounds.width + 16, bounds.height + 12, 8);
    graphics.strokeRoundedRect(bounds.x - 8, bounds.y - 6, bounds.width + 16, bounds.height + 12, 8);
    this.rootLayer?.add(graphics);
    this.rootLayer?.add(label);
  }

  private drawLegend(state: WorldState): void {
    const size = this.tileSize();
    const topLeft = this.toScreen([1, state.map.height - 2]);
    const text = `${state.map.width} x ${state.map.height} tiles | ${state.map.meters_per_tile}m/tile`;
    this.addLabel(text, topLeft.x + size * 4, topLeft.y + size * 0.45, 12, '#33413d');
  }

  private tileCenter(tile: Tile): { x: number; y: number } {
    const point = this.toScreen(tile);
    const half = this.tileSize() / 2;
    return { x: point.x + half, y: point.y + half };
  }

  private addLabel(text: string, x: number, y: number, fontSize: number, color = '#1f2828'): void {
    const scaledSize = Math.max(10, Math.round(fontSize * this.renderConfig.zoom));
    const label = this.add.text(x, y, text, {
      color,
      fontFamily: 'Inter, Arial, sans-serif',
      fontSize: `${scaledSize}px`,
      fontStyle: '600',
    });
    label.setOrigin(0.5, 0.5);
    this.rootLayer?.add(label);
  }
}

function objectSortWeight(object: SimulationObject): number {
  const weights: Record<string, number> = {
    conference_table: 1,
    chair: 2,
    door: 3,
    screen: 4,
    whiteboard: 4,
    light_switch: 5,
    laptop: 6,
    documents: 6,
    cup: 6,
    remote: 6,
  };
  return weights[object.type] ?? 99;
}

function shortObjectLabel(object: SimulationObject): string {
  const labels: Record<string, string> = {
    door: 'Door',
    screen: 'Screen',
    whiteboard: 'Whiteboard',
    light_switch: 'Light',
    laptop: 'Laptop',
    documents: 'Docs',
    cup: 'Cup',
    remote: 'Remote',
  };
  return labels[object.type] ?? object.label;
}

function agentColor(agent: AgentState): number {
  if (agent.public_mood === 'interrupted') {
    return 0xc95c4a;
  }
  const roleColors: Record<AgentState['role'], number> = {
    presenter: 0x2f6f9f,
    participant: 0x4f8f65,
    facilitator: 0x7b5fa8,
  };
  return roleColors[agent.role];
}

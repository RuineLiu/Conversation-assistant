// §8.1 视觉令牌
export const glass = {
  green: "#5dff7a",
  greenDim: "#2f8f43",
  greenGlow: "rgba(93,255,122,0.55)",
  bg: "#070a08",
  red: "#ff5f5f",
  amber: "#ffcf5d",
  mono: "'JetBrains Mono', monospace",
};

export const app = {
  bg: "#ffffff",
  text: "#1a1a1a",
  sub: "#8a8a8a",
  line: "#ececec",
  accent: "#000000",
  red: "#d94040",
  green: "#1f9d4d",
};

// HUD 背景：点阵 + 扫描线
export const hudBackground = {
  backgroundColor: glass.bg,
  backgroundImage: `radial-gradient(${glass.greenDim}22 1px, transparent 1px), repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(93,255,122,0.03) 3px, rgba(93,255,122,0.03) 4px)`,
  backgroundSize: "18px 18px, 100% 4px",
};

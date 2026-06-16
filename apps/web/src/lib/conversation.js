// 维护最近 N 句对话上下文
export function makeConversation(maxLines = 8) {
  const lines = [];
  return {
    push(text) { lines.push(text); if (lines.length > maxLines) lines.shift(); },
    context() { return [...lines]; },
    clear() { lines.length = 0; },
  };
}

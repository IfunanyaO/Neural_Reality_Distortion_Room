const ws = new WebSocket("ws://localhost:9000/ws");

document.getElementById("send").onclick = () => {
  const data = {
    distortion_intensity: parseFloat(
      document.getElementById("intensity").value
    ),
    shader_mode: document.getElementById("shader").value,
    style_prompt: document.getElementById("prompt").value,
  };
  ws.send(JSON.stringify(data));
};

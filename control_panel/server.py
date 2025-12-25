# control_panel/server.py

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from interaction.control_state import ControlState

app = FastAPI()
control_state = ControlState()

app.mount("/static", StaticFiles(directory="control_panel/static"), name="static")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[Artist Panel] WebSocket connected.")

    try:
        while True:
            data = await ws.receive_json()
            control_state.update(**data)
    except:
        print("[Artist Panel] WebSocket closed.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

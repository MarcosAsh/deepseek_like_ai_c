import modal

app = modal.App("deepseek-visual")

image = (
    modal.Image.debian_slim()
    .apt_install("cmake", "g++", "make")
    .add_local_dir(".", "/app", copy=True, ignore=[
        "build", "node_modules", "frontend/node_modules",
        "frontend/.next", ".git", "__pycache__", ".venv"
    ])
    .run_commands(
        "cd /app && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make node_server -j$(nproc)"
    )
)


@app.function(
    image=image,
    cpu=2,
    memory=512,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8080)
def server():
    import subprocess
    subprocess.Popen(["/app/build/node_server", "--port", "8080"], cwd="/app")

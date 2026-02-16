import modal

app = modal.App("deepseek-visual")

image = (
    modal.Image.debian_slim()
    .apt_install("cmake", "g++", "make")
    .copy_local_dir(".", "/app", ignore=[
        "build", "node_modules", "frontend/node_modules",
        "frontend/dist", ".git", "__pycache__"
    ])
    .run_commands(
        "cd /app && mkdir -p build && cd build && cmake .. && make node_server -j$(nproc)"
    )
)


@app.function(image=image)
@modal.web_server(port=8080)
def server():
    import subprocess
    subprocess.Popen(["/app/build/node_server", "--port", "8080"])

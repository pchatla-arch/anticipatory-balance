"""Render two saved, paired G1 trajectories with native MuJoCo and FFmpeg."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from foresight.envs.g1_impact.scene import build_scene


WIDTH, HEIGHT = 1060, 492
PANEL_WIDTH, PANEL_HEIGHT = 504, 298
FPS = 25
ARMS = ("ttc", "belief")
TITLES = {"ttc": "TTC threshold", "belief": "Lead-conditioned belief"}
INK, MUTED = "#ecf1ef", "#bac4c0"
BACKGROUND, CARD = "#17211f", "#27332f"


def fonts(directory):
    def face(name, size):
        return ImageFont.truetype(str(directory / name), size)

    return {
        "title": face("DejaVuSans-Bold.ttf", 30),
        "panel": face("DejaVuSans-Bold.ttf", 21),
        "label": face("DejaVuSans-Bold.ttf", 15),
        "text": face("DejaVuSans.ttf", 16),
        "time": face("DejaVuSansMono.ttf", 19),
    }


def validate(trace, record, model):
    steps, dt = record["steps"], record["dt_s"]
    if dt != 0.02 or steps != 300 or float(trace["dt"]) != dt:
        raise ValueError("Expected two 6 s encounters recorded at 50 Hz")
    if int(trace["seed"]) != record["seed"] or trace["scheduled_miss"].tolist() != [False, True]:
        raise ValueError("Scenario metadata differs from the recorded pair")
    for arm in ARMS:
        pose = trace[f"{arm}_qpos"]
        if pose.shape != (steps, 2, model.nq):
            raise ValueError(f"Pose shape differs from scene in {arm}")
        for encounter, entry in enumerate(record["arms"][arm]):
            terminal = int(trace[f"{arm}_term_step"][encounter])
            last = terminal if terminal >= 0 else steps - 1
            contacts = np.flatnonzero(trace[f"{arm}_impact_contact"][:last + 1, encounter])
            preparation = trace[f"{arm}_brace_active"][:last + 1, encounter]
            if (entry["scheduled_miss"] != bool(trace["scheduled_miss"][encounter])
                    or terminal != entry["term_step"]
                    or entry["contact_step"] != (int(contacts[0]) if len(contacts) else -1)
                    or not np.isclose(preparation.sum() * dt, entry["applied_brace_s"])
                    or not np.isfinite(pose[:last + 1, encounter]).all()):
                raise ValueError(f"Trace and record disagree for {arm}, encounter {encounter}")


def paint_frame(renderer, model, state, camera, trace, record, font, encounter, step):
    dt = record["dt_s"]
    frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(frame)
    title = ("Scheduled hit", "Scheduled near-miss")[encounter]
    draw.text((20, 12), title, font=font["title"], fill=INK)
    draw.text((21, 55), f"Encounter {encounter + 1} of 2  •  Paired G1 replay  •  100 ms detector delay",
              font=font["text"], fill=MUTED)
    draw.text((WIDTH - 20, 23), f"{(step + 1) * dt:.2f} / 6.00 s",
              font=font["time"], fill=INK, anchor="ra")
    for index, arm in enumerate(ARMS):
        x = 20 + index * 516
        entry = record["arms"][arm][encounter]
        terminal = entry["term_step"]
        ended = 0 <= terminal <= step
        row = min(step, terminal) if ended else step
        # Row t is the post-step pose. Draw the actual terminal row at and after failure.
        state.qpos[:] = trace[f"{arm}_qpos"][row, encounter]
        mujoco.mj_forward(model, state)
        renderer.update_scene(state, camera=camera)
        frame.paste(Image.fromarray(renderer.render()), (x, 135))

        draw.rectangle((x, 91, x + PANEL_WIDTH - 1, 134), fill=CARD)
        draw.text((x + 12, 100), TITLES[arm], font=font["panel"], fill=INK)
        active = bool(trace[f"{arm}_brace_active"][row, encounter])
        if ended:
            status, color = ("FALL" if entry["fell"] else "TERMINAL"), "#ffa98e"
        elif active:
            status, color = "BRACE", "#ffdc86"
        else:
            status, color = "WALK", "#b2e3c9"
        draw.text((x + PANEL_WIDTH - 12, 105), status, font=font["label"], fill=color, anchor="ra")
        contact = entry["contact_step"]
        first = (f"Contact {(contact + 1) * dt:.2f} s" if 0 <= contact <= row
                 else "No contact recorded")
        braced = dt * trace[f"{arm}_brace_active"][:row + 1, encounter].sum()
        second = (f"Terminal pose held at {(terminal + 1) * dt:.2f} s" if ended
                  else "Observer camera")
        draw.text((x, 442), f"{first}   •   Brace applied {braced:.2f} s",
                  font=font["label"], fill=INK)
        draw.text((x, 466), second, font=font["text"], fill=MUTED)
    return frame


def render(directory, font_directory):
    record = json.loads((directory / "example_record.json").read_text())
    font = fonts(font_directory)
    with tempfile.TemporaryDirectory() as tmp, np.load(directory / "example_traces.npz", allow_pickle=False) as trace:
        scene, _ = build_scene(Path(tmp))
        model = mujoco.MjModel.from_xml_path(str(scene))
        # Downsample only the visual shadow map so software rendering is inexpensive.
        model.vis.quality.shadowsize = 1024
        validate(trace, record, model)
        state = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        camera.elevation, camera.distance = -13, 3.0
        movie = directory / "paired-encounters.mp4"
        # A sibling temporary output permits atomic rename even across mounted filesystems.
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            stage = Path(staging)
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(stage / movie.name),
            ]
            with mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH) as renderer:
                process = subprocess.Popen(command, stdin=subprocess.PIPE)
                try:
                    for encounter, angle in enumerate((128, -155)):
                        # One fixed world camera per encounter, identical for both policies.
                        camera.azimuth = angle
                        camera.lookat[:] = ((-0.45, 0.60, 0.80), (1.02, -1.34, 0.80))[encounter]
                        for step in range(1, record["steps"], 2):
                            frame = paint_frame(renderer, model, state, camera, trace, record, font, encounter, step)
                            if encounter == 0 and step == 151:
                                frame.save(stage / "paired-encounters.jpg", quality=91, optimize=True)
                            process.stdin.write(frame.tobytes())
                finally:
                    process.stdin.close()
                    code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            (stage / movie.name).replace(movie)
            (stage / "paired-encounters.jpg").replace(directory / "paired-encounters.jpg")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=root / "docs/foresight/replay")
    parser.add_argument("--font-directory", type=Path, default=Path("/usr/share/fonts/truetype/dejavu"))
    args = parser.parse_args()
    render(args.directory, args.font_directory)

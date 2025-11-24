import datetime
import os
import shutil
import subprocess

import fastapi
import pandas as pd
from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from filelock import FileLock
from pydantic import BaseModel

app = fastapi.FastAPI()


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy")


class EvaluationRequest(BaseModel):
    repo_url: str
    revision: str
    track: str


class EvaluationResponse(BaseModel):
    success: bool = False
    error: str = ""
    score: float = -1


@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    url = request.repo_url.replace("https://github.com/", "git@github.com:") + ".git"
    revision = request.revision
    repo_name = url.split("/")[-1].split(".")[0]
    track = request.track
    template_name = "dl2025-flower-classification-competition"
    if template_name in repo_name:
        team = repo_name.replace(f"{template_name}-", "")
    else:
        team = repo_name

    print("-" * 20)
    now = datetime.datetime.now()
    print(f"[{now}] Evaluating {team} for {track} using {revision}")

    columns = ["team", "revision", "score", "timestamp"]
    with FileLock("scores.csv.lock"):
        if not os.path.exists("scores.csv"):
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.read_csv("scores.csv", index_col=0)

    team_subs = df[(df["track"] == track) & (df["team"] == team)]
    n_submissions = len(team_subs)
    if n_submissions > 4:
        return EvaluationResponse(
            error="You cannot submit more than 5 times to this track."
        )

    if not team_subs.empty:
        last_ts = datetime.datetime.fromisoformat(team_subs["timestamp"].max())
        time_diff = datetime.datetime.now() - last_ts
        cooldown = datetime.timedelta(hours=24)
        if time_diff < cooldown:
            wait_time = cooldown - time_diff
            hours, remainder = divmod(int(wait_time.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)
            return EvaluationResponse(
                error=f"You cannot submit more than once every 24 hours. Wait {hours}h {minutes}m."
            )

    wd = f"/home/ubuntu/actions-runner/_work/{repo_name}/{repo_name}"

    def clean() -> None:
        for item in os.listdir(wd):
            item = os.path.join(wd, item)
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
            except Exception as e:
                print(f"Failed to remove {item}: {e}")

    try:
        commands = [
            f"git clone {url} .",
            f"git checkout {revision}",
        ]
        result = subprocess.run(
            f"cd {wd} && " + " && ".join(commands), shell=True, check=True
        )

        files = [
            entry
            for entry in os.scandir(f"{wd}/models")
            if entry.is_file(follow_symlinks=False)
        ]
        if len(files) < 1:
            clean()
            return EvaluationResponse(error="You did not submit a model")
        if len(files) > 1:
            clean()
            return EvaluationResponse(error="You cannot submit more than 1 model")
        model_file_path = files[0].path

        commands = [
            "python3 -m venv .venv",
            ". .venv/bin/activate",
            "pip install -r requirements.txt "
            "--extra-index-url https://download.pytorch.org/whl/cpu",
        ]
        result = subprocess.run(
            f"cd {wd} && " + " && ".join(commands), shell=True, check=True
        )

        server_dir = "/home/ubuntu/competition-server"

        result = subprocess.run(
            f"cd {wd} && rm src/evaluate_model.py src/eval/evaluate.py src/eval/__init__.py",
            shell=True,
            check=True,
        )
        commands = [
            f"cp {server_dir}/evaluate_model.py src/",
            "touch src/eval/__init__.py",
            f"cp {server_dir}/evaluate.py src/eval/",
        ]
        result = subprocess.run(
            f"cd {wd} && " + " && ".join(commands), shell=True, check=True
        )

        commands = [
            ". .venv/bin/activate",
            "python -m src.evaluate_model -n",
        ]
        result = subprocess.run(
            f"cd {wd} && " + " && ".join(commands), shell=True, check=True
        )
        with open(f"{wd}/n_params.txt", "r") as f:
            n_trainable_params = int(f.readline())
        if track == "fast" and n_trainable_params > 100_000:
            clean()
            return EvaluationResponse(
                error="Models submitted to the `fast` track cannot"
                " have more than 100K trainable parameters. "
                f"Your model has {n_trainable_params}"
            )
        if track == "large" and n_trainable_params > 25_000_000:
            clean()
            return EvaluationResponse(
                error="Models submitted to the `fast` track cannot"
                " have more than 25M trainable parameters. "
                f"Your model has {n_trainable_params}"
            )

        commands = [
            ". .venv/bin/activate",
            'python -m src.evaluate_model -D "/home/ubuntu/competition-server/dataset/test"'
            + f" -p {model_file_path}",
        ]
        result = subprocess.run(
            f"cd {wd} && " + " && ".join(commands), shell=True, check=True
        )

        with open(f"{wd}/score.txt", "r") as f:
            score = f.readline()[:-1]

        clean()

        timestamp = datetime.datetime.now()
        df.loc[len(df)] = [team, track, revision, score, timestamp]
        df.to_csv("scores.csv")

        return EvaluationResponse(success=True, score=float(score))

    except Exception as e:
        clean()
        raise e


templates = Jinja2Templates(directory="templates")


def get_leaderboard_data(csv_path: str) -> dict:
    """
    Reads the CSV and processes it to build the leaderboard data.
    """
    try:
        # Read the CSV. Assumes the first column is the index.
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        return {"error": "scores.csv not found."}

    # Ensure score is numeric and timestamp is datetime
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["score", "timestamp"])

    leaderboards = {"fast": [], "large": []}

    # Sort by score so the first item for each group is the best
    df_sorted = df.sort_values(by="score", ascending=False)

    # Group by track and team
    grouped = df_sorted.groupby(["track", "team"])

    for (track, team), submissions in grouped:
        if track not in leaderboards:
            continue

        # The first submission is the best one (due to sorting)
        best_submission = submissions.iloc[0]

        # Get all other submissions (up to 4, max 5 total)
        other_submissions = submissions.iloc[1:5].to_dict("records")

        # Format data for the template
        team_data = {
            "team": team,
            "best_score": best_submission["score"],
            "best_revision": best_submission["revision"],
            # Format timestamp nicely
            "best_timestamp": best_submission["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "other_submissions": [],
        }

        # Format other submissions
        for sub in other_submissions:
            team_data["other_submissions"].append(
                {
                    "score": sub["score"],
                    "revision": sub["revision"],
                    "timestamp": sub["timestamp"].strftime("%Y-%m-%d %H:%M"),
                }
            )

        leaderboards[track].append(team_data)

    # Sort each leaderboard by the best score
    leaderboards["fast"] = sorted(
        leaderboards["fast"], key=lambda x: x["best_score"], reverse=True
    )
    leaderboards["large"] = sorted(
        leaderboards["large"], key=lambda x: x["best_score"], reverse=True
    )

    return leaderboards


@app.get("/", response_class=HTMLResponse)
def leaderboard(request: Request):
    """
    Serves the main leaderboard page.
    """
    # Process the data on each request
    leaderboard_data = get_leaderboard_data("scores.csv")

    if "error" in leaderboard_data:
        return f"<h1>Error</h1><p>{leaderboard_data['error']}</p>"

    return templates.TemplateResponse(
        "leaderboard.html", {"request": request, "leaderboards": leaderboard_data}
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("icons/icons8-flower-doodle-color-96.png")

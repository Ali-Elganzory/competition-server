import os
import shutil
import datetime
import subprocess

import fastapi
import pandas as pd
from fastapi import Request
from pydantic import BaseModel
from filelock import FileLock
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates


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

    columns = ["team", "revision", "score", "timestamp"]
    with FileLock("scores.csv.lock"):
        if not os.path.exists("scores.csv"):
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.read_csv("scores.csv", index_col=0)

    n_submissions = len(df[(df["track"] == track) & (df["team"] == repo_name)])
    if n_submissions > 4:
        return EvaluationResponse(error="You cannot submit more than 5 times to this track.")

    wd = f"/home/ubuntu/actions-runner/_work/{repo_name}/{repo_name}"

    commands = [
        f"git clone {url} .",
        f"git checkout {revision}",
    ]
    result = subprocess.run(f"cd {wd} && " + " && ".join(commands), shell=True)

    files = [entry.name for entry in os.scandir(f"{wd}/models") if entry.is_file(follow_symlinks=False)]
    if len(files) < 1:
        return EvaluationResponse(error="You did not submit a model")
    if len(files) > 1:
        return EvaluationResponse(error="You cannot submit more than 1 model")
    model_name = files[0]

    commands = [
        "python3 -m venv .venv",
        ". .venv/bin/activate",
        "pip install -r requirements.txt "
            "--extra-index-url https://download.pytorch.org/whl/cpu",
        "python -m src.dataset",
        "python -m src.evaluate_model",
    ]
    result = subprocess.run(f"cd {wd} && " + " && ".join(commands), shell=True)

    with open(f"{wd}/score.txt", "r") as f:
        score = f.readline()[:-1]

    for item in os.listdir(wd):
        item = os.path.join(wd, item)
        try:
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
        except Exception as e:
            print(f"Failed to remove {item}: {e}")

    timestamp = datetime.datetime.now()
    df.loc[len(df)] = [repo_name, track, revision, score, timestamp]
    df.to_csv("scores.csv")

    return EvaluationResponse(success=True, score=float(score))


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
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['score', 'timestamp'])

    leaderboards = {
        "fast": [],
        "large": []
    }

    # Sort by score so the first item for each group is the best
    df_sorted = df.sort_values(by='score', ascending=False)
    
    # Group by track and team
    grouped = df_sorted.groupby(['track', 'team'])

    for (track, team), submissions in grouped:
        if track not in leaderboards:
            continue
        
        # The first submission is the best one (due to sorting)
        best_submission = submissions.iloc[0]
        
        # Get all other submissions (up to 4, max 5 total)
        other_submissions = submissions.iloc[1:5].to_dict('records')

        # Format data for the template
        team_data = {
            "team": team,
            "best_score": best_submission['score'],
            "best_revision": best_submission['revision'],
            # Format timestamp nicely
            "best_timestamp": best_submission['timestamp'].strftime('%Y-%m-%d %H:%M'),
            "other_submissions": []
        }
        
        # Format other submissions
        for sub in other_submissions:
            team_data["other_submissions"].append({
                "score": sub['score'],
                "revision": sub['revision'],
                "timestamp": sub['timestamp'].strftime('%Y-%m-%d %H:%M')
            })

        leaderboards[track].append(team_data)

    # Sort each leaderboard by the best score
    leaderboards['fast'] = sorted(
        leaderboards['fast'], key=lambda x: x['best_score'], reverse=True
    )
    leaderboards['large'] = sorted(
        leaderboards['large'], key=lambda x: x['best_score'], reverse=True
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
        "leaderboard.html",
        {"request": request, "leaderboards": leaderboard_data}
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("icons/icons8-flower-doodle-color-96.png")
